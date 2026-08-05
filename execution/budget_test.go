package execution

import (
	"context"
	"errors"
	"testing"
)

func TestBudgetChargesAndSnapshots(t *testing.T) {
	t.Parallel()

	budget, err := NewBudget(10)
	if err != nil {
		t.Fatalf("NewBudget() error = %v", err)
	}
	if err := budget.Wait(context.Background(), 4); err != nil {
		t.Fatalf("Wait() error = %v", err)
	}
	snapshot := budget.Snapshot()
	if snapshot.Limit != 10 || snapshot.Spent != 4 || snapshot.Remaining != 6 {
		t.Fatalf("Snapshot() = %+v", snapshot)
	}
}

func TestBudgetRejectsOverspendWithoutCharging(t *testing.T) {
	t.Parallel()

	budget, err := NewBudget(2)
	if err != nil {
		t.Fatalf("NewBudget() error = %v", err)
	}
	if err := budget.Wait(context.Background(), 3); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("Wait() error = %v, want ErrBudgetExceeded", err)
	}
	if budget.Remaining() != 2 {
		t.Fatalf("Remaining() = %d, want 2", budget.Remaining())
	}
}

func TestChainStopsBeforeLaterLimiters(t *testing.T) {
	t.Parallel()

	budget, err := NewBudget(0)
	if err != nil {
		t.Fatalf("NewBudget() error = %v", err)
	}
	probe := &probeLimiter{}
	err = Chain(budget, probe).Wait(context.Background(), 1)
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("Wait() error = %v, want ErrBudgetExceeded", err)
	}
	if probe.calls != 0 {
		t.Fatalf("later limiter calls = %d, want 0", probe.calls)
	}
}

type probeLimiter struct {
	calls int
}

func (limiter *probeLimiter) Wait(context.Context, int) error {
	limiter.calls++
	return nil
}
