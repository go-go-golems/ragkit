package execution

import (
	"context"
	"fmt"

	"golang.org/x/sync/errgroup"
)

// Limiter gates work by resource units. Implementations may enforce a rate,
// finite budget, or another focused admission policy. They must honor context
// cancellation. A nil Limiter means no resource limit.
type Limiter interface {
	Wait(context.Context, int) error
}

// MapOptions controls bounded parallel work.
type MapOptions[T any] struct {
	// Workers is the maximum number of work functions running concurrently.
	// Values below one select one worker.
	Workers int

	// Limiter optionally gates each item before its work function starts. Use
	// Chain to combine a finite Budget with a TokenBucket rate.
	Limiter Limiter

	// Cost returns the resource units consumed by an item. Nil means one unit.
	// Costs must be positive.
	Cost func(T) int
}

// Map applies work to every item with bounded concurrency and optional resource
// rate limiting. Results preserve input order. The first error cancels pending
// work and is annotated with its input index.
func Map[T, R any](
	ctx context.Context,
	items []T,
	options MapOptions[T],
	work func(context.Context, T) (R, error),
) ([]R, error) {
	if work == nil {
		return nil, fmt.Errorf("work function is required")
	}
	if len(items) == 0 {
		return []R{}, nil
	}

	workers := options.Workers
	if workers < 1 {
		workers = 1
	}
	if workers > len(items) {
		workers = len(items)
	}

	type job struct {
		index int
		item  T
	}

	results := make([]R, len(items))
	jobs := make(chan job)
	group, groupContext := errgroup.WithContext(ctx)

	group.Go(func() error {
		defer close(jobs)
		for index, item := range items {
			select {
			case <-groupContext.Done():
				return groupContext.Err()
			case jobs <- job{index: index, item: item}:
			}
		}
		return nil
	})

	for range workers {
		group.Go(func() error {
			for current := range jobs {
				cost := 1
				if options.Cost != nil {
					cost = options.Cost(current.item)
				}
				if cost < 1 {
					return fmt.Errorf("item %d: resource cost must be positive", current.index)
				}
				if options.Limiter != nil {
					if err := options.Limiter.Wait(groupContext, cost); err != nil {
						return fmt.Errorf("item %d: wait for resources: %w", current.index, err)
					}
				}
				result, err := work(groupContext, current.item)
				if err != nil {
					return fmt.Errorf("item %d: %w", current.index, err)
				}
				results[current.index] = result
			}
			return nil
		})
	}

	if err := group.Wait(); err != nil {
		return nil, err
	}
	return results, nil
}
