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

func TestChainRollsBackEarlierReservationsWhenLaterLimiterRefuses(t *testing.T) {
	t.Parallel()

	budget, err := NewBudget(2)
	if err != nil {
		t.Fatalf("NewBudget() error = %v", err)
	}
	err = Chain(budget, rejectingLimiter{}).Wait(context.Background(), 1)
	if err == nil {
		t.Fatal("Wait() error = nil, want refusal")
	}
	if budget.Remaining() != 2 {
		t.Fatalf("Remaining() = %d, want rollback to 2", budget.Remaining())
	}
}

func TestChainCommitsReservationsAfterAllLimitersAccept(t *testing.T) {
	t.Parallel()

	budget, err := NewBudget(2)
	if err != nil {
		t.Fatalf("NewBudget() error = %v", err)
	}
	if err := Chain(budget, &probeLimiter{}).Wait(context.Background(), 1); err != nil {
		t.Fatalf("Wait() error = %v", err)
	}
	if budget.Remaining() != 1 {
		t.Fatalf("Remaining() = %d, want 1", budget.Remaining())
	}
}

func TestNestedChainPropagatesReservationCommitAndRollback(t *testing.T) {
	t.Parallel()

	accepted := &deferredLimiter{}
	if err := Chain(Chain(accepted)).Wait(t.Context(), 2); err != nil {
		t.Fatal(err)
	}
	if accepted.committed != 2 {
		t.Fatalf("committed units = %d, want 2", accepted.committed)
	}

	rolledBack := &deferredLimiter{}
	if err := Chain(Chain(rolledBack), rejectingLimiter{}).Wait(t.Context(), 3); err == nil {
		t.Fatal("Wait() error = nil, want refusal")
	}
	if rolledBack.committed != 0 {
		t.Fatalf("rolled-back committed units = %d, want 0", rolledBack.committed)
	}
}

type probeLimiter struct {
	calls int
}

type rejectingLimiter struct{}

type deferredLimiter struct {
	committed int
}

func (rejectingLimiter) Wait(context.Context, int) error {
	return errors.New("refused")
}

func (limiter *probeLimiter) Wait(context.Context, int) error {
	limiter.calls++
	return nil
}

func (limiter *deferredLimiter) Wait(ctx context.Context, units int) error {
	reservation, err := limiter.Reserve(ctx, units)
	if err != nil {
		return err
	}
	reservation.Commit()
	return nil
}

func (limiter *deferredLimiter) Reserve(_ context.Context, units int) (Reservation, error) {
	return newReservationWithCommit(func() { limiter.committed += units }, nil), nil
}
