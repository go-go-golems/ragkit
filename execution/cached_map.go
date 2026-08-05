package execution

import (
	"context"
	"fmt"
	"sync/atomic"
)

// CacheState describes the durable state of one input after MapCached returns.
type CacheState string

const (
	CacheHit     CacheState = "hit"
	CacheStored  CacheState = "stored"
	CachePending CacheState = "pending"
)

// CacheOutcome records the cache key and final state for one input.
type CacheOutcome struct {
	KeyDigest string     `json:"key_digest"`
	State     CacheState `json:"state"`
}

// CacheReport is suitable for an experiment observation.
type CacheReport struct {
	Hits      int            `json:"hits"`
	Misses    int            `json:"misses"`
	Writes    int            `json:"writes"`
	WorkCalls int            `json:"work_calls"`
	Outcomes  []CacheOutcome `json:"outcomes"`
}

// CachedMapOptions controls item identity, durable storage, and bounded work.
type CachedMapOptions[T any] struct {
	Map   MapOptions[T]
	Cache Cache
	Key   func(T) (Key, error)
}

// MapCached loads valid cached results before admitting misses to worker, rate,
// and budget controls. Every successful miss is stored immediately. Therefore,
// a later item failure leaves earlier completed items recoverable by the next
// invocation. Duplicate keys execute once and populate every matching input.
func MapCached[T, R any](
	ctx context.Context,
	items []T,
	options CachedMapOptions[T],
	work func(context.Context, T) (R, error),
	onResult ...func(int, R, CacheOutcome) error,
) ([]R, CacheReport, error) {
	report := CacheReport{Outcomes: make([]CacheOutcome, len(items))}
	if options.Cache == nil {
		return nil, report, fmt.Errorf("cache is required")
	}
	if options.Key == nil {
		return nil, report, fmt.Errorf("cache key function is required")
	}
	if work == nil {
		return nil, report, fmt.Errorf("work function is required")
	}
	if len(onResult) > 1 || (len(onResult) == 1 && onResult[0] == nil) {
		return nil, report, fmt.Errorf("at most one non-nil result callback is allowed")
	}
	notify := func(index int, value R) error {
		if len(onResult) == 0 {
			return nil
		}
		return onResult[0](index, value, report.Outcomes[index])
	}
	if len(items) == 0 {
		return []R{}, report, nil
	}

	type keyGroup struct {
		key     Key
		item    T
		indexes []int
	}
	groups := make([]keyGroup, 0, len(items))
	groupByDigest := make(map[string]int, len(items))
	for index, item := range items {
		key, err := options.Key(item)
		if err != nil {
			return nil, report, fmt.Errorf("item %d: build cache key: %w", index, err)
		}
		keyDigest, err := key.digest()
		if err != nil {
			return nil, report, fmt.Errorf("item %d: validate cache key: %w", index, err)
		}
		report.Outcomes[index] = CacheOutcome{KeyDigest: keyDigest, State: CachePending}
		if groupIndex, ok := groupByDigest[keyDigest]; ok {
			groups[groupIndex].indexes = append(groups[groupIndex].indexes, index)
			continue
		}
		groupByDigest[keyDigest] = len(groups)
		groups = append(groups, keyGroup{key: key, item: item, indexes: []int{index}})
	}

	results := make([]R, len(items))
	misses := make([]keyGroup, 0, len(groups))
	for _, group := range groups {
		var cached R
		found, err := options.Cache.Load(ctx, group.key, &cached)
		if err != nil {
			return nil, report, fmt.Errorf("load cache item %d: %w", group.indexes[0], err)
		}
		if !found {
			report.Misses += len(group.indexes)
			misses = append(misses, group)
			continue
		}
		for _, index := range group.indexes {
			results[index] = cached
			report.Outcomes[index].State = CacheHit
			report.Hits++
			if err := notify(index, cached); err != nil {
				return nil, report, fmt.Errorf("notify cached result %d: %w", index, err)
			}
		}
	}
	if len(misses) == 0 {
		return results, report, nil
	}

	type completed struct {
		value   R
		indexes []int
	}
	var writes atomic.Int64
	var workCalls atomic.Int64
	mapOptions := MapOptions[keyGroup]{
		Workers: options.Map.Workers,
		Limiter: options.Map.Limiter,
		Cost: func(group keyGroup) int {
			if options.Map.Cost == nil {
				return 1
			}
			return options.Map.Cost(group.item)
		},
	}
	completedMisses, err := Map(ctx, misses, mapOptions, func(ctx context.Context, group keyGroup) (completed, error) {
		workCalls.Add(1)
		value, err := work(ctx, group.item)
		if err != nil {
			return completed{}, err
		}
		// Once expensive work succeeds, finish its local atomic commit even if a
		// sibling item has just canceled the errgroup context.
		if err := options.Cache.Store(context.WithoutCancel(ctx), group.key, value); err != nil {
			return completed{}, fmt.Errorf("store cache result: %w", err)
		}
		writes.Add(1)
		for _, index := range group.indexes {
			report.Outcomes[index].State = CacheStored
			if err := notify(index, value); err != nil {
				return completed{}, fmt.Errorf("notify stored result %d: %w", index, err)
			}
		}
		return completed{value: value, indexes: group.indexes}, nil
	})
	report.Writes = int(writes.Load())
	report.WorkCalls = int(workCalls.Load())
	if err != nil {
		return nil, report, err
	}
	for _, result := range completedMisses {
		for _, index := range result.indexes {
			results[index] = result.value
		}
	}
	return results, report, nil
}
