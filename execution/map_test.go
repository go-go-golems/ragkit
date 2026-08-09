package execution

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestMapPreservesOrderAndWorkerLimit(t *testing.T) {
	t.Parallel()

	var active atomic.Int64
	var maximum atomic.Int64
	results, err := Map(
		context.Background(),
		[]int{5, 4, 3, 2, 1},
		MapOptions[int]{Workers: 2},
		func(ctx context.Context, value int) (int, error) {
			current := active.Add(1)
			defer active.Add(-1)
			for {
				observed := maximum.Load()
				if current <= observed || maximum.CompareAndSwap(observed, current) {
					break
				}
			}
			select {
			case <-ctx.Done():
				return 0, ctx.Err()
			case <-time.After(time.Duration(value) * time.Millisecond):
			}
			return value * 10, nil
		},
	)
	if err != nil {
		t.Fatalf("Map() error = %v", err)
	}
	if maximum.Load() > 2 {
		t.Fatalf("maximum active work = %d, want <= 2", maximum.Load())
	}
	for index, value := range results {
		want := (5 - index) * 10
		if value != want {
			t.Fatalf("results[%d] = %d, want %d", index, value, want)
		}
	}
}

func TestMapStopsWhenBudgetIsExceeded(t *testing.T) {
	t.Parallel()

	budget, err := NewBudget(3)
	if err != nil {
		t.Fatalf("NewBudget() error = %v", err)
	}
	_, err = Map(
		context.Background(),
		[]int{1, 1, 1, 1},
		MapOptions[int]{Workers: 1, Limiter: budget, Cost: func(value int) int { return value }},
		func(_ context.Context, value int) (int, error) { return value, nil },
	)
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("Map() error = %v, want ErrBudgetExceeded", err)
	}
	if budget.Spent() != 3 {
		t.Fatalf("budget spent = %d, want 3", budget.Spent())
	}
}

func TestMapRejectsNonPositiveCost(t *testing.T) {
	t.Parallel()

	_, err := Map(
		context.Background(),
		[]int{1},
		MapOptions[int]{Cost: func(int) int { return 0 }},
		func(_ context.Context, value int) (int, error) { return value, nil },
	)
	if err == nil {
		t.Fatal("Map() error = nil, want invalid cost")
	}
}
