package execution

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestTokenBucketBurstAndReplenishment(t *testing.T) {
	t.Parallel()

	limiter, err := NewTokenBucket(Rate{Units: 20, Per: time.Second, Burst: 1})
	if err != nil {
		t.Fatalf("NewTokenBucket() error = %v", err)
	}
	t.Cleanup(func() {
		if err := limiter.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})

	if err := limiter.Wait(context.Background(), 1); err != nil {
		t.Fatalf("first Wait() error = %v", err)
	}
	started := time.Now()
	if err := limiter.Wait(context.Background(), 1); err != nil {
		t.Fatalf("second Wait() error = %v", err)
	}
	if elapsed := time.Since(started); elapsed < 30*time.Millisecond {
		t.Fatalf("second Wait() elapsed = %s, want rate delay", elapsed)
	}
}

func TestTokenBucketHonorsCancellationAndRefunds(t *testing.T) {
	t.Parallel()

	limiter, err := NewTokenBucket(Rate{Units: 1, Per: time.Hour, Burst: 2})
	if err != nil {
		t.Fatalf("NewTokenBucket() error = %v", err)
	}
	t.Cleanup(func() {
		if err := limiter.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})

	if err := limiter.Wait(context.Background(), 1); err != nil {
		t.Fatalf("initial Wait() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := limiter.Wait(ctx, 2); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Wait() error = %v, want context.Canceled", err)
	}
	if err := limiter.Wait(context.Background(), 1); err != nil {
		t.Fatalf("Wait() after refund error = %v", err)
	}
}

func TestTokenBucketRejectsCostAboveBurst(t *testing.T) {
	t.Parallel()

	limiter, err := NewTokenBucket(Rate{Units: 1, Per: time.Second, Burst: 2})
	if err != nil {
		t.Fatalf("NewTokenBucket() error = %v", err)
	}
	t.Cleanup(func() {
		if err := limiter.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})

	if err := limiter.Wait(context.Background(), 3); err == nil {
		t.Fatal("Wait() error = nil, want burst error")
	}
}

func TestTokenBucketRejectsWaitAfterCloseWithBufferedTokens(t *testing.T) {
	t.Parallel()
	limiter, err := NewTokenBucket(Rate{Units: 1, Per: time.Hour, Burst: 2})
	if err != nil {
		t.Fatalf("NewTokenBucket() error = %v", err)
	}
	if err := limiter.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	for range 100 {
		if err := limiter.Wait(context.Background(), 1); err == nil {
			t.Fatal("Wait() error = nil after Close()")
		}
	}
}
