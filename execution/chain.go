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

func (chain limiterChain) Wait(ctx context.Context, units int) error {
	for index, limiter := range chain {
		if err := limiter.Wait(ctx, units); err != nil {
			return fmt.Errorf("limiter %d: %w", index, err)
		}
	}
	return nil
}
