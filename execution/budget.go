package execution

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// ErrBudgetExceeded identifies work rejected by a finite experiment budget.
var ErrBudgetExceeded = errors.New("resource budget exceeded")

// Budget is a concurrency-safe, non-replenishing resource budget. Successful
// admission consumes units permanently, including when the admitted work later
// fails. This models attempted provider cost and prevents retries from silently
// escaping the experiment ceiling.
type Budget struct {
	mu        sync.Mutex
	limit     int
	remaining int
}

var _ Limiter = (*Budget)(nil)
var _ ReservableLimiter = (*Budget)(nil)

// NewBudget creates a budget with total available resource units.
func NewBudget(total int) (*Budget, error) {
	if total < 0 {
		return nil, fmt.Errorf("budget total must not be negative")
	}
	return &Budget{limit: total, remaining: total}, nil
}

// Wait atomically charges units or returns ErrBudgetExceeded. It never blocks.
func (budget *Budget) Wait(ctx context.Context, units int) error {
	reservation, err := budget.Reserve(ctx, units)
	if err != nil {
		return err
	}
	reservation.Commit()
	return nil
}

// Reserve provisionally charges units so a composed limiter can refund them
// if a later admission policy refuses before work starts.
func (budget *Budget) Reserve(ctx context.Context, units int) (Reservation, error) {
	if budget == nil {
		return nil, fmt.Errorf("budget is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if units < 1 {
		return nil, fmt.Errorf("resource units must be positive")
	}

	budget.mu.Lock()
	defer budget.mu.Unlock()
	if units > budget.remaining {
		return nil, fmt.Errorf(
			"%w: requested %d, remaining %d, limit %d",
			ErrBudgetExceeded,
			units,
			budget.remaining,
			budget.limit,
		)
	}
	budget.remaining -= units
	return newReservation(func() {
		budget.mu.Lock()
		budget.remaining += units
		budget.mu.Unlock()
	}), nil
}

// Limit returns the initial budget.
func (budget *Budget) Limit() int {
	if budget == nil {
		return 0
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	return budget.limit
}

// Remaining returns currently unspent units.
func (budget *Budget) Remaining() int {
	if budget == nil {
		return 0
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	return budget.remaining
}

// Spent returns admitted resource units.
func (budget *Budget) Spent() int {
	if budget == nil {
		return 0
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	return budget.limit - budget.remaining
}

// BudgetSnapshot is suitable for experiment observations and reports.
type BudgetSnapshot struct {
	Limit     int `json:"limit"`
	Spent     int `json:"spent"`
	Remaining int `json:"remaining"`
}

// Snapshot atomically captures budget state.
func (budget *Budget) Snapshot() BudgetSnapshot {
	if budget == nil {
		return BudgetSnapshot{}
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	return BudgetSnapshot{
		Limit:     budget.limit,
		Spent:     budget.limit - budget.remaining,
		Remaining: budget.remaining,
	}
}
