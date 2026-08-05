package flow

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/go-go-golems/ragkit/execution"
)

// Bulk wraps a step whose provider call takes many items at once while every
// item's result stays individually durable — the embeddings shape (one
// request carries a batch of texts, one vector comes back per text, each
// vector caches under its own key). doBulk must return exactly one result
// per input item, in order.
//
// Cache hits never enter a bulk call; unique misses are grouped into batches
// of at most batchSize. Admission charges one unit per unique missed item;
// retry and classification follow the step's Policy around each bulk call —
// the retry coverage the embeddings path never had. A bulk-call failure
// honors the step's FailureMode for every item of the batch.
func Bulk[I, O any](s Step[I, O], doBulk func(context.Context, []I) ([]O, error), batchSize int) Step[I, O] {
	bulk := s
	bulk.Do = nil
	bulk.override = func(ctx context.Context, items []I, o Options) ([]Result[O], Report, error) {
		return runBulk(ctx, s, doBulk, batchSize, items, o)
	}
	return bulk
}

func runBulk[I, O any](
	ctx context.Context,
	s Step[I, O],
	doBulk func(context.Context, []I) ([]O, error),
	batchSize int,
	items []I,
	o Options,
) ([]Result[O], Report, error) {
	counts := StepReport{}
	meters := Meters{}
	var mutex sync.Mutex
	resources := make([]string, 0, len(s.Policy.Admission))
	for _, resource := range s.Policy.Admission {
		resources = append(resources, resource.Name)
	}
	report := func() Report {
		mutex.Lock()
		defer mutex.Unlock()
		snapshot := counts
		if len(meters) > 0 {
			snapshot.Meters = Meters{}
			snapshot.Meters.Add(meters)
		}
		snapshot.Spend = o.env.snapshot(resources)
		return Report{Steps: map[string]StepReport{s.Name: snapshot}}
	}
	fail := func(err error) ([]Result[O], Report, error) { return nil, report(), err }

	if doBulk == nil {
		return fail(fmt.Errorf("bulk step %q needs a bulk function", s.Name))
	}
	if batchSize < 1 {
		return fail(fmt.Errorf("bulk step %q batch size must be positive", s.Name))
	}
	if s.Name == "" {
		return fail(fmt.Errorf("every step needs a name"))
	}
	cached := s.Identity.Kind != "" && o.Store != nil
	counts.Items = len(items)

	// Group duplicate keys so one key executes once, and load hits before
	// admitting anything — cache hits are free.
	type keyGroup struct {
		key     execution.Key
		digest  string
		item    I
		indexes []int
	}
	results := make([]Result[O], len(items))
	groups := make([]keyGroup, 0, len(items))
	if cached {
		groupByDigest := make(map[string]int, len(items))
		for index, item := range items {
			key, err := s.Key(item)
			if err != nil {
				return fail(err)
			}
			digestValue, err := keyDigest(key)
			if err != nil {
				return fail(fmt.Errorf("step %q item %d: %w", s.Name, index, err))
			}
			if groupIndex, ok := groupByDigest[digestValue]; ok {
				groups[groupIndex].indexes = append(groups[groupIndex].indexes, index)
				continue
			}
			groupByDigest[digestValue] = len(groups)
			groups = append(groups, keyGroup{key: key, digest: digestValue, item: item, indexes: []int{index}})
		}
	} else {
		for index, item := range items {
			groups = append(groups, keyGroup{item: item, indexes: []int{index}})
		}
	}

	notify := func(index int, value O, outcome execution.CacheOutcome) error {
		if s.OnResult == nil {
			return nil
		}
		if err := s.OnResult(ctx, index, value, outcome); err != nil {
			return fmt.Errorf("step %q item %d: result hook: %w", s.Name, index, err)
		}
		return nil
	}

	misses := make([]keyGroup, 0, len(groups))
	for _, group := range groups {
		if !cached {
			misses = append(misses, group)
			continue
		}
		var loaded O
		found, err := o.Store.Load(ctx, group.key, &loaded)
		if err != nil {
			return fail(fmt.Errorf("step %q item %d: load cache entry: %w", s.Name, group.indexes[0], err))
		}
		if !found {
			counts.Misses += len(group.indexes)
			misses = append(misses, group)
			continue
		}
		outcome := execution.CacheOutcome{KeyDigest: group.digest, State: execution.CacheHit}
		for _, index := range group.indexes {
			counts.Hits++
			results[index] = Result[O]{Value: loaded, Cache: outcome}
			if o.Ledger != nil {
				if err := o.Ledger.Event(ctx, Event{Step: s.Name, Index: index, Type: EventHit}); err != nil {
					return fail(fmt.Errorf("step %q: ledger event: %w", s.Name, err))
				}
			}
			if err := notify(index, loaded, outcome); err != nil {
				return fail(err)
			}
		}
	}
	if len(misses) == 0 {
		return results, report(), nil
	}

	type batch struct{ groups []keyGroup }
	batches := make([]batch, 0, (len(misses)+batchSize-1)/batchSize)
	for start := 0; start < len(misses); start += batchSize {
		end := min(start+batchSize, len(misses))
		batches = append(batches, batch{groups: misses[start:end]})
	}

	progressStop := make(chan struct{})
	defer close(progressStop)
	go func() {
		ticker := time.NewTicker(progressInterval)
		defer ticker.Stop()
		for {
			select {
			case <-progressStop:
				return
			case <-ticker.C:
				for name, step := range report().Steps {
					log.Info().
						Str("step", name).
						Int("items", step.Items).
						Int("hits", step.Hits).
						Int("stored", step.Stored).
						Int("work_calls", step.WorkCalls).
						Int("retries", step.Retries).
						Msg("flow bulk progress")
				}
			}
		}
	}()

	classifier := s.Policy.Retry.Class
	if classifier == nil {
		classifier = DefaultClassifier
	}
	attempts := s.Policy.Retry.Attempts
	if attempts < 1 {
		attempts = 1
	}
	backoff := s.Policy.Retry.Backoff.withDefaults()
	count := func(update func(*StepReport)) {
		mutex.Lock()
		update(&counts)
		mutex.Unlock()
	}

	_, err := execution.Map(ctx, batches, execution.MapOptions[batch]{
		Workers: max(s.Policy.Workers, 1),
	}, func(ctx context.Context, current batch) (struct{}, error) {
		units := len(current.groups)
		for _, name := range resources {
			if err := o.env.limiter(name).Wait(ctx, units); err != nil {
				if errors.Is(err, execution.ErrBudgetExceeded) {
					plan := o.env.plan(name)
					return struct{}{}, fmt.Errorf(
						"step %q: resource %q admission refused a %d-item batch: %w; stated ceiling %d, admitted budget %d — cache hits are free",
						s.Name, name, units, err, plan.Ceiling, plan.Budget,
					)
				}
				return struct{}{}, fmt.Errorf("step %q: wait for resource %q: %w", s.Name, name, err)
			}
		}

		batchItems := make([]I, len(current.groups))
		for position, group := range current.groups {
			batchItems[position] = group.item
		}
		delay := backoff.Base
		var values []O
		var lastErr error
		var lastClass ErrorClass
		for attempt := 1; attempt <= attempts; attempt++ {
			if attempt > 1 {
				jittered := delay + time.Duration(rand.Int63n(int64(delay)/2+1))
				select {
				case <-ctx.Done():
					return struct{}{}, ctx.Err()
				case <-time.After(jittered):
				}
				delay *= 2
				if delay > backoff.Cap {
					delay = backoff.Cap
				}
			}
			returned, err := doBulk(ctx, batchItems)
			count(func(counts *StepReport) { counts.WorkCalls++ })
			if err == nil {
				if len(returned) != len(batchItems) {
					return struct{}{}, fmt.Errorf(
						"step %q: bulk call returned %d results for %d items",
						s.Name, len(returned), len(batchItems),
					)
				}
				values = returned
				lastErr = nil
				break
			}
			lastErr = err
			if ctx.Err() != nil {
				return struct{}{}, fmt.Errorf("step %q: %w", s.Name, err)
			}
			lastClass = classifier.Classify(err)
			log.Debug().
				Str("step", s.Name).
				Int("attempt", attempt).
				Int("batch_items", units).
				Str("class", lastClass.String()).
				Err(err).
				Msg("flow bulk call failed")
			if lastClass == Transient && attempt < attempts {
				count(func(counts *StepReport) {
					counts.Retries++
					if counts.RetriesByClass == nil {
						counts.RetriesByClass = map[string]int{}
					}
					counts.RetriesByClass[lastClass.String()]++
				})
				continue
			}
			break
		}
		if lastErr != nil {
			mode := s.Policy.OnError
			if lastClass == Fatal || mode == FailFast {
				return struct{}{}, fmt.Errorf("step %q batch [%s]: %w", s.Name, lastClass, lastErr)
			}
			mutex.Lock()
			for _, group := range current.groups {
				for _, index := range group.indexes {
					if mode == Skip {
						counts.Skipped++
						results[index] = Result[O]{Skipped: true}
						continue
					}
					counts.Quarantined++
					results[index] = Result[O]{Quarantined: &ItemError{
						Step: s.Name, Index: index, Class: lastClass,
						Attempts: attempts, Message: lastErr.Error(),
					}}
				}
			}
			mutex.Unlock()
			return struct{}{}, nil
		}

		for position, group := range current.groups {
			value := values[position]
			outcome := execution.CacheOutcome{}
			if cached {
				// Commit each result even if a sibling batch just canceled
				// the run: completed provider work stays recoverable.
				if err := o.Store.Store(context.WithoutCancel(ctx), group.key, value); err != nil {
					return struct{}{}, fmt.Errorf("step %q item %d: store cache entry: %w", s.Name, group.indexes[0], err)
				}
				count(func(counts *StepReport) { counts.Stored++ })
				outcome = execution.CacheOutcome{KeyDigest: group.digest, State: execution.CacheStored}
			}
			if s.Meter != nil {
				metered := s.Meter(value)
				mutex.Lock()
				meters.Add(metered)
				mutex.Unlock()
			}
			mutex.Lock()
			for _, index := range group.indexes {
				results[index] = Result[O]{Value: value, Cache: outcome}
			}
			mutex.Unlock()
			for _, index := range group.indexes {
				if o.Ledger != nil {
					if err := o.Ledger.Event(ctx, Event{Step: s.Name, Index: index, Type: EventStored}); err != nil {
						return struct{}{}, fmt.Errorf("step %q: ledger event: %w", s.Name, err)
					}
				}
				if err := notify(index, value, outcome); err != nil {
					return struct{}{}, err
				}
			}
		}
		return struct{}{}, nil
	})
	if err != nil {
		return fail(err)
	}
	return results, report(), nil
}
