package flow

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pkg/errors"
	"github.com/stretchr/testify/require"

	"github.com/go-go-golems/ragkit/execution"
)

// doubler is a cached step used throughout: value -> value*2.
func doubler(name string, policy Policy) Step[int, int] {
	return Step[int, int]{
		Name: name,
		Identity: Identity[int]{
			Kind:    "test-" + name,
			Version: "v1",
			Key:     func(value int) ([]byte, error) { return []byte(strconv.Itoa(value)), nil },
		},
		Policy: policy,
		Do: func(_ context.Context, value int) (int, error) {
			return value * 2, nil
		},
	}
}

func fastRetry(attempts int) RetrySpec {
	return RetrySpec{Attempts: attempts, Backoff: Backoff{Base: time.Millisecond, Cap: 2 * time.Millisecond}}
}

func TestRetryArithmeticSaturatesAtCap(t *testing.T) {
	cap := time.Duration(math.MaxInt64)
	require.Equal(t, cap, nextRetryDelay(cap-1, cap))
	require.Equal(t, cap, nextRetryDelay(cap/2+1, cap))
	require.Equal(t, 2*time.Second, nextRetryDelay(time.Second, cap))
	require.Equal(t, time.Duration(0), nextRetryDelay(time.Second, 0))

	for range 100 {
		delay := retryDelay(cap, cap)
		require.GreaterOrEqual(t, delay, time.Duration(0))
		require.LessOrEqual(t, delay, cap)
	}
}

func TestRunPreservesOrderUnderSkewedLatencies(t *testing.T) {
	step := Step[int, string]{
		Name: "skewed",
		Do: func(_ context.Context, value int) (string, error) {
			// Earlier items sleep longer, so completion order inverts input
			// order unless results are position-aligned.
			time.Sleep(time.Duration(20-value) * time.Millisecond)
			return strconv.Itoa(value), nil
		},
		Policy: Policy{Workers: 8},
	}
	items := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	results, report, err := Run(context.Background(), step, items, Options{})
	require.NoError(t, err)
	require.Len(t, results, len(items))
	for index, item := range items {
		require.Equal(t, strconv.Itoa(item), results[index].Value)
	}
	require.Equal(t, len(items), report.Step("skewed").Items)
	require.Equal(t, len(items), report.Step("skewed").WorkCalls)
}

func TestRunCachesHitsMissesAndReplays(t *testing.T) {
	store := NewMemoryStore()
	var calls atomic.Int64
	step := doubler("cache", Policy{Workers: 4})
	base := step.Do
	step.Do = func(ctx context.Context, value int) (int, error) {
		calls.Add(1)
		return base(ctx, value)
	}

	first, report, err := Run(context.Background(), step, []int{1, 2, 3}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, int64(3), calls.Load())
	require.Equal(t, 3, report.Step("cache").Misses)
	require.Equal(t, 3, report.Step("cache").WorkCalls)
	require.Equal(t, 0, report.Step("cache").Hits)
	for index, result := range first {
		require.Equal(t, (index+1)*2, result.Value)
		require.Equal(t, execution.CacheStored, result.Cache.State)
		require.NotEmpty(t, result.Cache.KeyDigest)
	}

	// Replay: same items, zero fresh work.
	second, report, err := Run(context.Background(), step, []int{1, 2, 3}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, int64(3), calls.Load(), "replay must be free")
	require.Equal(t, 3, report.Step("cache").Hits)
	require.Equal(t, 0, report.Step("cache").WorkCalls)
	for index, result := range second {
		require.Equal(t, first[index].Value, result.Value)
		require.Equal(t, execution.CacheHit, result.Cache.State)
		require.Equal(t, first[index].Cache.KeyDigest, result.Cache.KeyDigest)
	}
}

// The durable mechanism is swappable: the same step replays identically
// against the on-disk FileCache and the in-memory store.
func TestRunStoreIsSwappable(t *testing.T) {
	fileCache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	require.NoError(t, err)
	for _, store := range []Store{fileCache, NewMemoryStore()} {
		step := doubler("swap", Policy{Workers: 2})
		_, first, err := Run(context.Background(), step, []int{7, 8}, Options{Store: store})
		require.NoError(t, err)
		require.Equal(t, 2, first.Step("swap").Misses)
		_, second, err := Run(context.Background(), step, []int{7, 8}, Options{Store: store})
		require.NoError(t, err)
		require.Equal(t, 2, second.Step("swap").Hits)
		require.Equal(t, 0, second.Step("swap").WorkCalls)
	}
}

func TestRunUncachedWhenStoreNilOrIdentityEmpty(t *testing.T) {
	var calls atomic.Int64
	step := doubler("uncached", Policy{})
	base := step.Do
	step.Do = func(ctx context.Context, value int) (int, error) {
		calls.Add(1)
		return base(ctx, value)
	}
	// Nil store: cached identity, uncached run.
	_, _, err := Run(context.Background(), step, []int{1}, Options{})
	require.NoError(t, err)
	// Empty kind: pure compute even with a store.
	pure := step
	pure.Identity = Identity[int]{}
	store := NewMemoryStore()
	_, _, err = Run(context.Background(), pure, []int{1}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, int64(2), calls.Load())
	require.Equal(t, 0, store.Len())
}

func TestRunDeduplicatesIdenticalKeysInFlight(t *testing.T) {
	var calls atomic.Int64
	step := Step[int, int]{
		Name: "dedup",
		Identity: Identity[int]{
			Kind:    "test-dedup",
			Version: "v1",
			Key:     func(value int) ([]byte, error) { return []byte(strconv.Itoa(value)), nil },
		},
		Policy: Policy{Workers: 8},
		Do: func(_ context.Context, value int) (int, error) {
			calls.Add(1)
			time.Sleep(10 * time.Millisecond)
			return value * 2, nil
		},
	}
	items := []int{5, 5, 5, 5, 5, 5}
	results, report, err := Run(context.Background(), step, items, Options{Store: NewMemoryStore()})
	require.NoError(t, err)
	require.Equal(t, int64(1), calls.Load(), "one key must execute once per process")
	for _, result := range results {
		require.Equal(t, 10, result.Value)
	}
	require.Equal(t, 1, report.Step("dedup").WorkCalls)
	require.Equal(t, 6, report.Step("dedup").Items)
}

func TestRunRetriesTransientErrorsWithCounts(t *testing.T) {
	var calls atomic.Int64
	step := Step[int, int]{
		Name:   "flaky",
		Policy: Policy{Retry: fastRetry(4)},
		Do: func(_ context.Context, value int) (int, error) {
			if calls.Add(1) < 3 {
				return 0, errors.New("read: connection timed out")
			}
			return value, nil
		},
	}
	results, report, err := Run(context.Background(), step, []int{42}, Options{})
	require.NoError(t, err)
	require.Equal(t, 42, results[0].Value)
	require.Equal(t, 2, report.Step("flaky").Retries)
	require.Equal(t, map[string]int{"transient": 2}, report.Step("flaky").RetriesByClass)
	require.Equal(t, 3, report.Step("flaky").WorkCalls)
}

func TestRunChargesEveryRetryAttemptAgainstAdmission(t *testing.T) {
	var calls atomic.Int64
	step := Step[int, int]{
		Name: "budgeted-retry",
		Policy: Policy{
			Retry:     fastRetry(3),
			Admission: []Resource{{Name: "provider-calls", Ceiling: 3, Budget: 1}},
		},
		Do: func(_ context.Context, _ int) (int, error) {
			calls.Add(1)
			return 0, errors.New("unexpected EOF")
		},
	}
	_, report, err := Run(context.Background(), step, []int{1}, Options{})
	require.ErrorIs(t, err, execution.ErrBudgetExceeded)
	require.Equal(t, int64(1), calls.Load(), "a refused retry must not invoke provider work")
	require.Equal(t, 1, report.Step("budgeted-retry").WorkCalls)
	require.Equal(t, execution.BudgetSnapshot{Limit: 1, Spent: 1, Remaining: 0}, report.Step("budgeted-retry").Spend["provider-calls"])
}

func TestRunDoesNotRetryFatalOrCancellation(t *testing.T) {
	var calls atomic.Int64
	fatal := Step[int, int]{
		Name:   "fatal",
		Policy: Policy{Retry: fastRetry(5)},
		Do: func(_ context.Context, _ int) (int, error) {
			calls.Add(1)
			return 0, errors.New("status=400: bad request")
		},
	}
	_, _, err := Run(context.Background(), fatal, []int{1}, Options{})
	require.Error(t, err)
	require.Equal(t, int64(1), calls.Load(), "fatal errors must not retry")

	calls.Store(0)
	canceled := Step[int, int]{
		Name:   "canceled",
		Policy: Policy{Retry: fastRetry(5)},
		Do: func(ctx context.Context, _ int) (int, error) {
			calls.Add(1)
			return 0, errors.Wrap(context.Canceled, "stream")
		},
	}
	_, _, err = Run(context.Background(), canceled, []int{1}, Options{})
	require.Error(t, err)
	require.Equal(t, int64(1), calls.Load(), "cancellation must never retry")
}

func TestRunQuarantineTurnsItemErrorsIntoRecords(t *testing.T) {
	step := Step[int, int]{
		Name:   "judge-like",
		Policy: Policy{Workers: 2, OnError: Quarantine, Retry: fastRetry(2)},
		Do: func(_ context.Context, value int) (int, error) {
			if value == 2 {
				return 0, AsDataError(errors.New("malformed verdict JSON"))
			}
			return value * 10, nil
		},
	}
	results, report, err := Run(context.Background(), step, []int{1, 2, 3}, Options{})
	require.NoError(t, err, "a quarantined item must not fail the run")
	require.Equal(t, 10, results[0].Value)
	require.Equal(t, 30, results[2].Value)
	require.Nil(t, results[0].Quarantined)
	record := results[1].Quarantined
	require.NotNil(t, record)
	require.Equal(t, "judge-like", record.Step)
	require.Equal(t, 1, record.Index)
	require.Equal(t, DataError, record.Class)
	require.Equal(t, 1, record.Attempts)
	require.Contains(t, record.Message, "malformed verdict JSON")
	require.Equal(t, 1, report.Step("judge-like").Quarantined)
	require.Equal(t, 0, results[1].Value)
}

func TestRunQuarantineNeverStoresBadItems(t *testing.T) {
	store := NewMemoryStore()
	step := doubler("poison", Policy{OnError: Quarantine})
	step.Do = func(_ context.Context, value int) (int, error) {
		return 0, AsDataError(errors.New("bad response"))
	}
	results, _, err := Run(context.Background(), step, []int{1}, Options{Store: store})
	require.NoError(t, err)
	require.NotNil(t, results[0].Quarantined)
	require.Equal(t, 0, store.Len(), "quarantined items must not poison the cache")
}

func TestRunFatalClassKillsRunEvenUnderQuarantine(t *testing.T) {
	step := Step[int, int]{
		Name:   "hard-fail",
		Policy: Policy{OnError: Quarantine},
		Do: func(_ context.Context, _ int) (int, error) {
			return 0, errors.New("status=401: bad api key")
		},
	}
	_, _, err := Run(context.Background(), step, []int{1}, Options{})
	require.Error(t, err, "fatal errors always fail the run")
	require.Contains(t, err.Error(), "hard-fail")
}

func TestRunFailFastCancelsPendingWork(t *testing.T) {
	var calls atomic.Int64
	step := Step[int, int]{
		Name:   "failfast",
		Policy: Policy{Workers: 1},
		Do: func(_ context.Context, value int) (int, error) {
			calls.Add(1)
			if value == 0 {
				return 0, errors.New("deterministic failure")
			}
			return value, nil
		},
	}
	_, _, err := Run(context.Background(), step, []int{0, 1, 2, 3, 4, 5, 6, 7}, Options{})
	require.Error(t, err)
	require.Contains(t, err.Error(), `step "failfast" item 0`)
	require.Less(t, calls.Load(), int64(8), "first error should cancel pending work")
}

func TestRunSkipDropsWithVisibleCount(t *testing.T) {
	step := Step[int, int]{
		Name:   "skipper",
		Policy: Policy{OnError: Skip},
		Do: func(_ context.Context, value int) (int, error) {
			if value == 1 {
				return 0, AsDataError(errors.New("drop me"))
			}
			return value, nil
		},
	}
	results, report, err := Run(context.Background(), step, []int{0, 1, 2}, Options{})
	require.NoError(t, err)
	require.True(t, results[1].Skipped)
	require.False(t, results[0].Skipped)
	require.Equal(t, 1, report.Step("skipper").Skipped)
}

func TestRunPreflightRefusalStatesArithmetic(t *testing.T) {
	step := doubler("expensive", Policy{
		Admission: []Resource{{Name: "expensive-calls", Ceiling: 240, Budget: 100}},
	})
	_, _, err := Run(context.Background(), step, []int{1}, Options{
		Preflight: &Preflight{MaxEstimatedUSD: 100, AllowUnpriced: true},
	})
	require.Error(t, err)
	require.Equal(t,
		`step "expensive": resource "expensive-calls" budget 100 cannot cover the stated ceiling of 240 calls; raise the budget to at least 240 (cache hits are free)`,
		err.Error(),
	)
}

func TestRunPreflightMonetaryGate(t *testing.T) {
	unit := 0.02
	step := doubler("priced", Policy{
		Admission: []Resource{{Name: "priced-calls", Ceiling: 1000, Budget: 1000, UnitUSD: &unit}},
	})
	_, _, err := Run(context.Background(), step, []int{1}, Options{
		Preflight: &Preflight{MaxEstimatedUSD: 5},
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "estimated provider cost 20.000000 exceeds maximum USD 5.000000")

	// Unpriced resources refuse unless explicitly allowed.
	unpriced := doubler("unpriced", Policy{
		Admission: []Resource{{Name: "unpriced-calls", Ceiling: 10, Budget: 10}},
	})
	_, _, err = Run(context.Background(), unpriced, []int{1}, Options{Preflight: &Preflight{MaxEstimatedUSD: 5}})
	require.Error(t, err)
	require.Contains(t, err.Error(), "pricing unavailable for unpriced-calls")
}

func TestRunRejectsNonFiniteUnitPriceBeforeWork(t *testing.T) {
	for _, unit := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		var calls atomic.Int64
		step := doubler("priced", Policy{Admission: []Resource{{
			Name: "priced-calls", Ceiling: 1, Budget: 1, UnitUSD: &unit,
		}}})
		base := step.Do
		step.Do = func(ctx context.Context, value int) (int, error) {
			calls.Add(1)
			return base(ctx, value)
		}
		_, _, err := Run(t.Context(), step, []int{1}, Options{})
		require.ErrorContains(t, err, "finite and non-negative")
		require.Zero(t, calls.Load())
	}
}

func TestRunRejectsInvalidPreflightMaximumBeforeWork(t *testing.T) {
	for _, maximum := range []float64{-1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		var calls atomic.Int64
		step := doubler("invalid-gate", Policy{})
		base := step.Do
		step.Do = func(ctx context.Context, value int) (int, error) {
			calls.Add(1)
			return base(ctx, value)
		}
		_, _, err := Run(t.Context(), step, []int{1}, Options{Preflight: &Preflight{MaxEstimatedUSD: maximum}})
		require.ErrorContains(t, err, "maximum USD must be finite and non-negative")
		require.Zero(t, calls.Load())
	}
}

func TestRunMultiResourceAdmissionRollsBackEarlierBudgets(t *testing.T) {
	resources := []Resource{
		{Name: "first", Ceiling: 1, Budget: 1},
		{Name: "refuses", Ceiling: 1, Budget: 0},
	}
	step := doubler("transactional", Policy{Admission: resources})
	shared := Options{}.Share()
	_, _, err := Run(t.Context(), step, []int{1}, shared)
	require.ErrorIs(t, err, execution.ErrBudgetExceeded)
	require.Equal(t, 0, shared.Snapshots()["first"].Spent)
}

func TestRunRejectsDuplicateAdmissionResourcesBeforeWork(t *testing.T) {
	calls := 0
	step := doubler("duplicate", Policy{Admission: []Resource{
		{Name: "calls", Ceiling: 2, Budget: 2},
		{Name: "calls", Ceiling: 2, Budget: 2},
	}})
	step.Do = func(context.Context, int) (int, error) { calls++; return 0, nil }
	_, _, err := Run(t.Context(), step, []int{1}, Options{})
	require.ErrorContains(t, err, "more than once")
	require.Zero(t, calls)
}

func TestRunRejectsUnknownFailureModeBeforeWork(t *testing.T) {
	calls := 0
	step := doubler("unknown-mode", Policy{OnError: FailureMode(99)})
	step.Do = func(context.Context, int) (int, error) { calls++; return 0, nil }
	_, _, err := Run(t.Context(), step, []int{1}, Options{})
	require.ErrorContains(t, err, "unknown failure mode")
	require.Zero(t, calls)
}

func TestRunSharedPreflightKeepsRefusingUnpricedPlan(t *testing.T) {
	var calls atomic.Int64
	step := doubler("unpriced", Policy{
		Admission: []Resource{{Name: "unpriced-calls", Ceiling: 1, Budget: 1}},
	})
	base := step.Do
	step.Do = func(ctx context.Context, value int) (int, error) {
		calls.Add(1)
		return base(ctx, value)
	}
	options := Options{Preflight: &Preflight{MaxEstimatedUSD: 1}}.Share()

	for range 2 {
		_, _, err := Run(context.Background(), step, []int{1}, options)
		require.ErrorContains(t, err, "pricing unavailable for unpriced-calls")
	}
	require.Zero(t, calls.Load())
}

func TestDeclareRefusalDoesNotCommitPartialState(t *testing.T) {
	price := 1.0
	options := Options{Preflight: &Preflight{MaxEstimatedUSD: 1}}.Share()
	err := options.Declare("too-expensive",
		Resource{Name: "first", Ceiling: 1, Budget: 1, UnitUSD: &price},
		Resource{Name: "second", Ceiling: 1, Budget: 1, UnitUSD: &price},
	)
	require.ErrorContains(t, err, "exceeds maximum")
	require.Empty(t, options.Snapshots())
	require.Zero(t, options.Cost().EstimatedUSD)

	require.NoError(t, options.Declare("valid", Resource{
		Name: "first", Ceiling: 1, Budget: 1, UnitUSD: &price,
	}))
	require.Contains(t, options.Snapshots(), "first")
}

func TestRunSharedResourcePlansCompareUnitPriceValues(t *testing.T) {
	firstPrice := 0.02
	secondPrice := 0.02
	first := doubler("first", Policy{Admission: []Resource{{
		Name: "priced-calls", Ceiling: 2, Budget: 2, UnitUSD: &firstPrice,
	}}})
	second := doubler("second", Policy{Admission: []Resource{{
		Name: "priced-calls", Ceiling: 2, Budget: 2, UnitUSD: &secondPrice,
	}}})
	options := Options{Preflight: &Preflight{MaxEstimatedUSD: 1}}.Share()

	_, _, err := Run(context.Background(), first, []int{1}, options)
	require.NoError(t, err)
	_, _, err = Run(context.Background(), second, []int{2}, options)
	require.NoError(t, err)
}

func TestRetryDelayNeverExceedsCap(t *testing.T) {
	for range 100 {
		require.LessOrEqual(t, retryDelay(10*time.Millisecond, 10*time.Millisecond), 10*time.Millisecond)
		require.LessOrEqual(t, retryDelay(20*time.Millisecond, 10*time.Millisecond), 10*time.Millisecond)
	}
}

func TestRunBudgetRefusalMidRunIsFatalWithWording(t *testing.T) {
	step := Step[int, int]{
		Name: "budgeted",
		Policy: Policy{
			Workers:   1,
			Admission: []Resource{{Name: "budgeted-calls", Ceiling: 3, Budget: 2}},
		},
		Do: func(_ context.Context, value int) (int, error) { return value, nil },
	}
	_, _, err := Run(context.Background(), step, []int{1, 2, 3}, Options{})
	require.Error(t, err)
	require.ErrorIs(t, err, execution.ErrBudgetExceeded)
	require.Contains(t, err.Error(), `resource "budgeted-calls" admission refused`)
	require.Contains(t, err.Error(), "stated ceiling 3, admitted budget 2")
	require.Contains(t, err.Error(), "cache hits are free")
}

func TestRunCacheHitsAreFreeOfAdmission(t *testing.T) {
	store := NewMemoryStore()
	step := doubler("free-hits", Policy{
		Admission: []Resource{{Name: "free-hits-calls", Ceiling: 3, Budget: 3}},
	})
	shared := Options{Store: store}.Share()
	_, _, err := Run(context.Background(), step, []int{1, 2, 3}, shared)
	require.NoError(t, err)
	// The shared budget is fully spent; a replay must still succeed because
	// hits bypass admission entirely.
	results, report, err := Run(context.Background(), step, []int{1, 2, 3}, shared)
	require.NoError(t, err)
	require.Equal(t, 3, report.Step("free-hits").Hits)
	require.Equal(t, 6, results[2].Value)
}

func TestRunSharedOptionsShareBudgetsAcrossRuns(t *testing.T) {
	resource := Resource{Name: "shared-calls", Ceiling: 2, Budget: 2}
	build := func(name string) Step[int, int] {
		return Step[int, int]{
			Name:   name,
			Policy: Policy{Admission: []Resource{resource}},
			Do:     func(_ context.Context, value int) (int, error) { return value, nil },
		}
	}
	shared := Options{}.Share()
	_, _, err := Run(context.Background(), build("first"), []int{1, 2}, shared)
	require.NoError(t, err)
	_, _, err = Run(context.Background(), build("second"), []int{3}, shared)
	require.Error(t, err, "the second step draws from the same budget")
	require.ErrorIs(t, err, execution.ErrBudgetExceeded)

	// Unshared options rebuild budgets per run.
	_, _, err = Run(context.Background(), build("third"), []int{1, 2}, Options{})
	require.NoError(t, err)
}

func TestRunConflictingResourceDeclarationsRefuse(t *testing.T) {
	shared := Options{}.Share()
	stepA := doubler("a", Policy{Admission: []Resource{{Name: "calls", Ceiling: 2, Budget: 2}}})
	stepB := doubler("b", Policy{Admission: []Resource{{Name: "calls", Ceiling: 9, Budget: 9}}})
	_, _, err := Run(context.Background(), stepA, nil, shared)
	require.NoError(t, err)
	_, _, err = Run(context.Background(), stepB, nil, shared)
	require.Error(t, err)
	require.Contains(t, err.Error(), `resource "calls" is declared differently`)
}

func TestRunRateLimiterComposesAfterBudget(t *testing.T) {
	var waited atomic.Int64
	rate := limiterFunc(func(ctx context.Context, units int) error {
		waited.Add(int64(units))
		return nil
	})
	step := doubler("rated", Policy{
		Admission: []Resource{{Name: "rated-calls", Ceiling: 2, Budget: 2}},
	})
	_, _, err := Run(context.Background(), step, []int{1, 2}, Options{
		Store: NewMemoryStore(),
		Rates: map[string]execution.Limiter{"rated-calls": rate},
	})
	require.NoError(t, err)
	require.Equal(t, int64(2), waited.Load())
}

type limiterFunc func(context.Context, int) error

func (f limiterFunc) Wait(ctx context.Context, units int) error { return f(ctx, units) }

func TestRunMetersCountFreshWorkOnly(t *testing.T) {
	store := NewMemoryStore()
	step := doubler("metered", Policy{})
	step.Meter = func(value int) Meters { return Meters{"tokens": float64(value)} }
	_, report, err := Run(context.Background(), step, []int{1, 2}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, Meters{"tokens": 6}, report.Step("metered").Meters)
	_, report, err = Run(context.Background(), step, []int{1, 2}, Options{Store: store})
	require.NoError(t, err)
	require.Nil(t, report.Step("metered").Meters, "cache hits are not metered")
}

func TestRunAttemptMeterCountsFailedAttempts(t *testing.T) {
	step := Step[int, int]{
		Name: "failed-meter", Policy: Policy{Retry: RetrySpec{
			Attempts: 2, Backoff: fastRetry(2).Backoff,
			Class: ClassifierFunc(func(error) ErrorClass { return Transient }),
		}},
		Do:           func(context.Context, int) (int, error) { return 5, errors.New("provider failed") },
		AttemptMeter: func(value int, _ error) Meters { return Meters{"tokens": float64(value)} },
	}
	_, report, err := Run(t.Context(), step, []int{1}, Options{})
	require.Error(t, err)
	require.Equal(t, Meters{"tokens": 10}, report.Step("failed-meter").Meters)
}

func TestRunRejectsNegativeRetryAttempts(t *testing.T) {
	step := doubler("negative-retry", Policy{Retry: RetrySpec{Attempts: -1}})
	_, _, err := Run(t.Context(), step, []int{1}, Options{})
	require.ErrorContains(t, err, "retry attempts must not be negative")
}

func TestRunRejectsUnknownClassifierResult(t *testing.T) {
	step := Step[int, int]{
		Name:   "bad-classifier",
		Policy: Policy{Retry: RetrySpec{Attempts: 1, Class: ClassifierFunc(func(error) ErrorClass { return ErrorClass(99) })}},
		Do:     func(context.Context, int) (int, error) { return 0, errors.New("failed") },
	}
	_, _, err := Run(t.Context(), step, []int{1}, Options{})
	require.ErrorContains(t, err, "unknown error class 99")
}

func TestRunReportsSpendSnapshots(t *testing.T) {
	step := doubler("spender", Policy{
		Admission: []Resource{{Name: "spender-calls", Ceiling: 2, Budget: 5}},
	})
	_, report, err := Run(context.Background(), step, []int{1, 2}, Options{})
	require.NoError(t, err)
	require.Equal(t, execution.BudgetSnapshot{Limit: 5, Spent: 2, Remaining: 3}, report.Step("spender").Spend["spender-calls"])
}

type recordingLedger struct {
	mutex  sync.Mutex
	events []Event
	fail   bool
}

func (ledger *recordingLedger) Event(_ context.Context, event Event) error {
	ledger.mutex.Lock()
	defer ledger.mutex.Unlock()
	if ledger.fail {
		return errors.New("journal disk full")
	}
	ledger.events = append(ledger.events, event)
	return nil
}

func (ledger *recordingLedger) byType(eventType EventType) []Event {
	ledger.mutex.Lock()
	defer ledger.mutex.Unlock()
	var matched []Event
	for _, event := range ledger.events {
		if event.Type == eventType {
			matched = append(matched, event)
		}
	}
	return matched
}

func TestRunLedgerReceivesEvents(t *testing.T) {
	store := NewMemoryStore()
	ledger := &recordingLedger{}
	step := doubler("journaled", Policy{})
	_, _, err := Run(context.Background(), step, []int{1, 2}, Options{Store: store, Ledger: ledger})
	require.NoError(t, err)
	require.Len(t, ledger.byType(EventStored), 2)
	_, _, err = Run(context.Background(), step, []int{1, 2}, Options{Store: store, Ledger: ledger})
	require.NoError(t, err)
	require.Len(t, ledger.byType(EventHit), 2)
	for _, event := range ledger.events {
		require.Equal(t, "journaled", event.Step)
	}
}

func TestRunLedgerFailureFailsTheRun(t *testing.T) {
	ledger := &recordingLedger{fail: true}
	step := doubler("journaled", Policy{})
	_, _, err := Run(context.Background(), step, []int{1}, Options{Store: NewMemoryStore(), Ledger: ledger})
	require.Error(t, err)
	require.Contains(t, err.Error(), "ledger event")
}

func TestRunEmptyItems(t *testing.T) {
	step := doubler("empty", Policy{})
	results, report, err := Run(context.Background(), step, nil, Options{})
	require.NoError(t, err)
	require.Empty(t, results)
	require.Equal(t, 0, report.Step("empty").Items)
}

func TestRunFlowStoreInteroperatesWithMapCached(t *testing.T) {
	// The same FileCache entry must serve both the legacy MapCached path
	// and flow.Run: flow is wiring, identity is contract.
	directory := t.TempDir()
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: directory})
	require.NoError(t, err)

	type payload struct {
		Text string `json:"text"`
	}
	keyOf := func(value int) (execution.Key, error) {
		return execution.NewKey("interop", "v1", map[string]int{"value": value})
	}
	_, _, err = execution.MapCached(context.Background(), []int{1, 2},
		execution.CachedMapOptions[int]{
			Map:   execution.MapOptions[int]{Workers: 1},
			Cache: cache,
			Key:   keyOf,
		},
		func(_ context.Context, value int) (payload, error) {
			return payload{Text: fmt.Sprintf("value-%d", value)}, nil
		},
	)
	require.NoError(t, err)

	step := Step[int, payload]{
		Name: "interop",
		Identity: Identity[int]{
			Kind:    "interop",
			Version: "v1",
			Key: func(value int) ([]byte, error) {
				return []byte(fmt.Sprintf(`{"value":%d}`, value)), nil
			},
		},
		Do: func(_ context.Context, _ int) (payload, error) {
			return payload{}, errors.New("must not be called: every item is cached")
		},
	}
	results, report, err := Run(context.Background(), step, []int{1, 2}, Options{Store: cache})
	require.NoError(t, err)
	require.Equal(t, 2, report.Step("interop").Hits)
	require.Equal(t, payload{Text: "value-1"}, results[0].Value)
	require.Equal(t, payload{Text: "value-2"}, results[1].Value)
}
