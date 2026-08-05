package execution

import (
	"context"
	"fmt"
	"sync/atomic"
)

// CachedBatchMapOptions controls per-item cache identity and bounded batch
// execution. BatchSize limits the number of unique cache misses passed to one
// work call. Limiter admission charges the sum of the item costs in a batch.
type CachedBatchMapOptions[T any] struct {
	Workers   int
	Limiter   Limiter
	Cost      func(T) int
	BatchSize int
	Cache     Cache
	Key       func(T) (Key, error)
}

// MapCachedBatches loads and stores individual results while executing unique
// cache misses in bounded batches. Results preserve input order, duplicate keys
// execute once, and each successful result is stored separately before the next
// result is committed. A later batch or store failure therefore leaves all
// earlier durable items recoverable.
func MapCachedBatches[T, R any](
	ctx context.Context,
	items []T,
	options CachedBatchMapOptions[T],
	work func(context.Context, []T) ([]R, error),
) ([]R, CacheReport, error) {
	report := CacheReport{Outcomes: make([]CacheOutcome, len(items))}
	if options.Cache == nil {
		return nil, report, fmt.Errorf("cache is required")
	}
	if options.Key == nil {
		return nil, report, fmt.Errorf("cache key function is required")
	}
	if work == nil {
		return nil, report, fmt.Errorf("batch work function is required")
	}
	if options.BatchSize < 1 {
		return nil, report, fmt.Errorf("batch size must be positive")
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
		}
	}
	if len(misses) == 0 {
		return results, report, nil
	}

	type batch struct {
		groups []keyGroup
		cost   int
	}
	batches := make([]batch, 0, (len(misses)+options.BatchSize-1)/options.BatchSize)
	for start := 0; start < len(misses); start += options.BatchSize {
		end := min(start+options.BatchSize, len(misses))
		current := batch{groups: misses[start:end]}
		for _, group := range current.groups {
			itemCost := 1
			if options.Cost != nil {
				itemCost = options.Cost(group.item)
			}
			if itemCost < 1 {
				return nil, report, fmt.Errorf(
					"item %d: resource cost must be positive",
					group.indexes[0],
				)
			}
			current.cost += itemCost
		}
		batches = append(batches, current)
	}

	type completed struct {
		groups []keyGroup
		values []R
	}
	var writes atomic.Int64
	var workCalls atomic.Int64
	completedBatches, err := Map(ctx, batches, MapOptions[batch]{
		Workers: options.Workers,
		Limiter: options.Limiter,
		Cost:    func(current batch) int { return current.cost },
	}, func(ctx context.Context, current batch) (completed, error) {
		workCalls.Add(1)
		batchItems := make([]T, len(current.groups))
		for index, group := range current.groups {
			batchItems[index] = group.item
		}
		values, err := work(ctx, batchItems)
		if err != nil {
			return completed{}, err
		}
		if len(values) != len(current.groups) {
			return completed{}, fmt.Errorf(
				"batch work returned %d results for %d items",
				len(values),
				len(current.groups),
			)
		}
		for index, group := range current.groups {
			if err := options.Cache.Store(context.WithoutCancel(ctx), group.key, values[index]); err != nil {
				return completed{}, fmt.Errorf("store cache result %d: %w", group.indexes[0], err)
			}
			writes.Add(1)
			for _, inputIndex := range group.indexes {
				report.Outcomes[inputIndex].State = CacheStored
			}
		}
		return completed{groups: current.groups, values: values}, nil
	})
	report.Writes = int(writes.Load())
	report.WorkCalls = int(workCalls.Load())
	if err != nil {
		return nil, report, err
	}
	for _, current := range completedBatches {
		for groupIndex, group := range current.groups {
			for _, inputIndex := range group.indexes {
				results[inputIndex] = current.values[groupIndex]
			}
		}
	}
	return results, report, nil
}
