package execution

import (
	"context"
	"fmt"
)

// Chain returns a Limiter that applies each non-nil limiter in order. Put a
// Budget before a TokenBucket to reject unaffordable work without waiting for a
// rate token.
func Chain(limiters ...Limiter) Limiter {
	filtered := make([]Limiter, 0, len(limiters))
	for _, limiter := range limiters {
		if limiter != nil {
			filtered = append(filtered, limiter)
		}
	}
	return limiterChain(filtered)
}

type limiterChain []Limiter

var _ Limiter = limiterChain(nil)
var _ ReservableLimiter = limiterChain(nil)

func (chain limiterChain) Wait(ctx context.Context, units int) error {
	reservation, err := chain.Reserve(ctx, units)
	if err != nil {
		return err
	}
	reservation.Commit()
	return nil
}

// Reserve provisionally admits work through every limiter in the chain. A
// chain is itself reservable, so larger admission transactions can safely
// compose per-resource budget-and-rate chains without committing an earlier
// resource when a later resource refuses the same work.
func (chain limiterChain) Reserve(ctx context.Context, units int) (Reservation, error) {
	reservations := make([]Reservation, 0, len(chain))
	rollback := func() {
		for index := len(reservations) - 1; index >= 0; index-- {
			reservations[index].Rollback()
		}
	}
	for index, limiter := range chain {
		if reservable, ok := limiter.(ReservableLimiter); ok {
			reservation, err := reservable.Reserve(ctx, units)
			if err != nil {
				rollback()
				return nil, fmt.Errorf("limiter %d: %w", index, err)
			}
			reservations = append(reservations, reservation)
			continue
		}
		if err := limiter.Wait(ctx, units); err != nil {
			rollback()
			return nil, fmt.Errorf("limiter %d: %w", index, err)
		}
	}
	commit := func() {
		for _, reservation := range reservations {
			reservation.Commit()
		}
	}
	return newReservationWithCommit(commit, rollback), nil
}
