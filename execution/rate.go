package execution

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Rate defines a token-bucket resource rate. Units are replenished evenly over
// Per, up to Burst. For example, Units=10, Per=time.Second, Burst=20 permits a
// sustained 10 units/second and a burst of 20 units.
type Rate struct {
	Units int
	Per   time.Duration
	Burst int
}

// TokenBucket is a process-local resource limiter. Close it when the experiment
// no longer needs it.
type TokenBucket struct {
	tokens    chan struct{}
	done      chan struct{}
	closeOnce sync.Once
	closed    atomic.Bool
	wg        sync.WaitGroup
}

var _ Limiter = (*TokenBucket)(nil)
var _ ReservableLimiter = (*TokenBucket)(nil)

// NewTokenBucket creates a full token bucket and starts its replenisher.
func NewTokenBucket(rate Rate) (*TokenBucket, error) {
	if rate.Units < 1 {
		return nil, fmt.Errorf("rate units must be positive")
	}
	if rate.Per <= 0 {
		return nil, fmt.Errorf("rate period must be positive")
	}
	if rate.Burst < 1 {
		return nil, fmt.Errorf("rate burst must be positive")
	}

	interval := rate.Per / time.Duration(rate.Units)
	if interval <= 0 {
		return nil, fmt.Errorf("rate interval is below timer resolution")
	}

	limiter := &TokenBucket{
		tokens: make(chan struct{}, rate.Burst),
		done:   make(chan struct{}),
	}
	for range rate.Burst {
		limiter.tokens <- struct{}{}
	}

	limiter.wg.Add(1)
	go limiter.replenish(interval)
	return limiter, nil
}

func (limiter *TokenBucket) replenish(interval time.Duration) {
	defer limiter.wg.Done()

	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-limiter.done:
			return
		case <-ticker.C:
			select {
			case limiter.tokens <- struct{}{}:
			default:
			}
		}
	}
}

// Wait acquires units from the bucket. If the context is canceled after a
// partial acquisition, already-acquired units are returned.
func (limiter *TokenBucket) Wait(ctx context.Context, units int) error {
	reservation, err := limiter.Reserve(ctx, units)
	if err != nil {
		return err
	}
	reservation.Commit()
	return nil
}

// Reserve acquires tokens provisionally for transactional limiter chains.
func (limiter *TokenBucket) Reserve(ctx context.Context, units int) (Reservation, error) {
	if limiter == nil {
		return nil, fmt.Errorf("token bucket is nil")
	}
	if units < 1 {
		return nil, fmt.Errorf("resource units must be positive")
	}
	if units > cap(limiter.tokens) {
		return nil, fmt.Errorf("resource units %d exceed burst %d", units, cap(limiter.tokens))
	}
	if limiter.closed.Load() {
		return nil, fmt.Errorf("token bucket is closed")
	}

	acquired := 0
	for acquired < units {
		if limiter.closed.Load() {
			limiter.refund(acquired)
			return nil, fmt.Errorf("token bucket is closed")
		}
		select {
		case <-ctx.Done():
			limiter.refund(acquired)
			return nil, ctx.Err()
		case <-limiter.done:
			limiter.refund(acquired)
			return nil, fmt.Errorf("token bucket is closed")
		case <-limiter.tokens:
			acquired++
			if limiter.closed.Load() {
				limiter.refund(acquired)
				return nil, fmt.Errorf("token bucket is closed")
			}
		}
	}
	return newReservation(func() { limiter.refund(units) }), nil
}

func (limiter *TokenBucket) refund(units int) {
	for range units {
		select {
		case limiter.tokens <- struct{}{}:
		default:
			return
		}
	}
}

// Close stops token replenishment and unblocks waiters.
func (limiter *TokenBucket) Close() error {
	if limiter == nil {
		return nil
	}
	limiter.closeOnce.Do(func() {
		limiter.closed.Store(true)
		close(limiter.done)
		limiter.wg.Wait()
	})
	return nil
}
