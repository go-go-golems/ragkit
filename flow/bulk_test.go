package flow

import (
	"context"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/pkg/errors"
	"github.com/stretchr/testify/require"

	"github.com/go-go-golems/ragkit/execution"
)

func bulkStep(name string) Step[int, int] {
	return Step[int, int]{
		Name: name,
		Identity: Identity[int]{
			Kind:    "test-" + name,
			Version: "v1",
			Key:     func(value int) ([]byte, error) { return []byte(strconv.Itoa(value)), nil },
		},
		Policy: Policy{Workers: 2},
	}
}

func doubleAll(calls *atomic.Int64, batches *[][]int, mutex *sync.Mutex) func(context.Context, []int) ([]int, error) {
	return func(_ context.Context, batch []int) ([]int, error) {
		if calls != nil {
			calls.Add(1)
		}
		if batches != nil {
			mutex.Lock()
			*batches = append(*batches, append([]int(nil), batch...))
			mutex.Unlock()
		}
		out := make([]int, len(batch))
		for index, value := range batch {
			out[index] = value * 2
		}
		return out, nil
	}
}

func TestBulkBatchesMissesAndCachesPerItem(t *testing.T) {
	store := NewMemoryStore()
	var calls atomic.Int64
	var batches [][]int
	var mutex sync.Mutex
	step := Bulk(bulkStep("embed"), doubleAll(&calls, &batches, &mutex), 3)

	items := []int{1, 2, 3, 4, 5, 6, 7}
	results, report, err := Run(context.Background(), step, items, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, int64(3), calls.Load(), "7 misses in batches of 3")
	for index, item := range items {
		require.Equal(t, item*2, results[index].Value)
		require.Equal(t, execution.CacheStored, results[index].Cache.State)
	}
	require.Equal(t, 7, report.Step("embed").Misses)
	require.Equal(t, 7, store.Len(), "every item durable under its own key")

	// Replay: all hits, no bulk calls.
	results, report, err = Run(context.Background(), step, items, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, int64(3), calls.Load())
	require.Equal(t, 7, report.Step("embed").Hits)
	require.Equal(t, execution.CacheHit, results[0].Cache.State)

	// Partial replay: only the two new items enter a bulk call.
	calls.Store(0)
	mutex.Lock()
	batches = nil
	mutex.Unlock()
	_, report, err = Run(context.Background(), step, []int{1, 2, 8, 9}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, int64(1), calls.Load())
	mutex.Lock()
	require.Equal(t, [][]int{{8, 9}}, batches)
	mutex.Unlock()
	require.Equal(t, 2, report.Step("embed").Hits)
	require.Equal(t, 2, report.Step("embed").Misses)
}

func TestBulkUncachedResultsEmitDone(t *testing.T) {
	ledger := &recordingLedger{}
	step := Bulk(bulkStep("uncached"), doubleAll(nil, nil, nil), 2)
	step.Identity = Identity[int]{}

	_, _, err := Run(context.Background(), step, []int{1, 2}, Options{Ledger: ledger})
	require.NoError(t, err)
	require.Len(t, ledger.byType(EventDone), 2)
	require.Empty(t, ledger.byType(EventStored))
}

func TestBulkRejectsUnknownClassifierResult(t *testing.T) {
	base := bulkStep("bad-classifier")
	base.Policy.Retry = RetrySpec{Attempts: 1, Class: ClassifierFunc(func(error) ErrorClass { return ErrorClass(99) })}
	step := Bulk(base, func(context.Context, []int) ([]int, error) { return nil, errors.New("failed") }, 2)
	_, _, err := Run(t.Context(), step, []int{1}, Options{})
	require.ErrorContains(t, err, "unknown error class 99")
}

func TestBulkDeduplicatesKeysAndChargesUniqueMisses(t *testing.T) {
	budget := Resource{Name: "embed-items", Ceiling: 2, Budget: 2}
	base := bulkStep("dedup-bulk")
	base.Policy.Admission = []Resource{budget}
	var calls atomic.Int64
	step := Bulk(base, doubleAll(&calls, nil, nil), 10)
	// Four inputs, two unique keys: the budget of 2 must suffice.
	results, report, err := Run(context.Background(), step, []int{5, 5, 6, 6}, Options{Store: NewMemoryStore()})
	require.NoError(t, err)
	require.Equal(t, int64(1), calls.Load())
	require.Equal(t, 10, results[0].Value)
	require.Equal(t, 12, results[3].Value)
	require.Equal(t, execution.BudgetSnapshot{Limit: 2, Spent: 2, Remaining: 0}, report.Step("dedup-bulk").Spend["embed-items"])
}

func TestBulkRetriesTransientBatchFailures(t *testing.T) {
	base := bulkStep("flaky-bulk")
	base.Policy.Retry = fastRetry(3)
	var calls atomic.Int64
	step := Bulk(base, func(_ context.Context, batch []int) ([]int, error) {
		if calls.Add(1) < 2 {
			return nil, errors.New("unexpected EOF")
		}
		out := make([]int, len(batch))
		for index, value := range batch {
			out[index] = value * 2
		}
		return out, nil
	}, 10)
	ledger := &recordingLedger{}
	results, report, err := Run(context.Background(), step, []int{1, 2}, Options{Store: NewMemoryStore(), Ledger: ledger})
	require.NoError(t, err, "the retry the embeddings incident never had")
	require.Equal(t, 2, results[0].Value)
	require.Equal(t, 1, report.Step("flaky-bulk").Retries)
	require.Equal(t, 2, report.Step("flaky-bulk").WorkCalls)
	retries := ledger.byType(EventRetry)
	require.Len(t, retries, 2)
	for index, event := range retries {
		require.Equal(t, index, event.Index)
		require.Equal(t, Transient.String(), event.Class)
		require.Equal(t, 1, event.Attempt)
		require.Contains(t, event.Error, "unexpected EOF")
	}
}

func TestBulkChargesEveryRetryAttemptAgainstAdmission(t *testing.T) {
	base := bulkStep("budgeted-bulk-retry")
	base.Policy.Retry = fastRetry(3)
	base.Policy.Admission = []Resource{{Name: "embedding-items", Ceiling: 6, Budget: 2}}
	var calls atomic.Int64
	step := Bulk(base, func(_ context.Context, _ []int) ([]int, error) {
		calls.Add(1)
		return nil, errors.New("unexpected EOF")
	}, 10)
	_, report, err := Run(context.Background(), step, []int{1, 2}, Options{Store: NewMemoryStore()})
	require.ErrorIs(t, err, execution.ErrBudgetExceeded)
	require.Equal(t, int64(1), calls.Load(), "a refused bulk retry must not invoke provider work")
	require.Equal(t, 1, report.Step("budgeted-bulk-retry").WorkCalls)
	require.Equal(t, execution.BudgetSnapshot{Limit: 2, Spent: 2, Remaining: 0}, report.Step("budgeted-bulk-retry").Spend["embedding-items"])
}

func TestBulkAttemptMeterCountsReturnedValuesOnEveryAttempt(t *testing.T) {
	base := bulkStep("metered-bulk-retry")
	base.Policy.Retry = fastRetry(2)
	base.AttemptMeter = func(value int, _ error) Meters { return Meters{"tokens": float64(value)} }
	var calls atomic.Int64
	step := Bulk(base, func(_ context.Context, batch []int) ([]int, error) {
		values := []int{3, 5}
		if calls.Add(1) == 1 {
			return values, errors.New("unexpected EOF")
		}
		return values[:len(batch)], nil
	}, 10)
	_, report, err := Run(t.Context(), step, []int{1, 2}, Options{})
	require.NoError(t, err)
	require.Equal(t, Meters{"tokens": 16}, report.Step("metered-bulk-retry").Meters)
}

func TestBulkRetryLedgerFailurePropagates(t *testing.T) {
	base := bulkStep("retry-ledger-failure")
	base.Policy.Retry = fastRetry(2)
	step := Bulk(base, func(context.Context, []int) ([]int, error) {
		return nil, errors.New("unexpected EOF")
	}, 10)

	_, _, err := Run(t.Context(), step, []int{1}, Options{Ledger: &recordingLedger{fail: true}})
	require.ErrorContains(t, err, "ledger event")
}

func TestBulkFailureModes(t *testing.T) {
	broken := func(_ context.Context, _ []int) ([]int, error) {
		return nil, AsDataError(errors.New("bulk response malformed"))
	}
	failFast := Bulk(bulkStep("bulk-failfast"), broken, 2)
	_, _, err := Run(context.Background(), failFast, []int{1, 2, 3}, Options{Store: NewMemoryStore()})
	require.Error(t, err)

	base := bulkStep("bulk-quarantine")
	base.Policy.OnError = Quarantine
	quarantine := Bulk(base, broken, 2)
	store := NewMemoryStore()
	results, report, err := Run(context.Background(), quarantine, []int{1, 2, 3}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, 3, report.Step("bulk-quarantine").Quarantined)
	require.NotNil(t, results[2].Quarantined)
	require.Equal(t, 2, results[2].Quarantined.Index)
	require.Equal(t, 1, results[2].Quarantined.Attempts)
	require.Equal(t, 0, store.Len(), "quarantined batches must not poison the cache")
}

type rejectingStore struct{}

func (rejectingStore) Load(context.Context, execution.Key, any) (bool, error) { return false, nil }
func (rejectingStore) Store(context.Context, execution.Key, any) error {
	return errors.New("simulated store failure")
}

func TestBulkMetersCompletedProviderWorkBeforeStoreFailure(t *testing.T) {
	base := bulkStep("meter-before-store")
	base.Meter = func(value int) Meters { return Meters{"tokens": float64(value)} }
	step := Bulk(base, doubleAll(nil, nil, nil), 10)
	_, report, err := Run(t.Context(), step, []int{1, 2}, Options{Store: rejectingStore{}})
	require.ErrorContains(t, err, "simulated store failure")
	require.Equal(t, Meters{"tokens": 6}, report.Step("meter-before-store").Meters)
}

func TestBulkMultiResourceAdmissionRollsBackEarlierBudgets(t *testing.T) {
	base := bulkStep("transactional-bulk")
	base.Policy.Admission = []Resource{
		{Name: "bulk-first", Ceiling: 2, Budget: 2},
		{Name: "bulk-refuses", Ceiling: 2, Budget: 1},
	}
	shared := Options{}.Share()
	_, _, err := Run(t.Context(), Bulk(base, doubleAll(nil, nil, nil), 2), []int{1, 2}, shared)
	require.ErrorIs(t, err, execution.ErrBudgetExceeded)
	require.Equal(t, 0, shared.Snapshots()["bulk-first"].Spent)
}

func TestBulkTerminalOutcomesAreJournaledAndLedgerFailuresPropagate(t *testing.T) {
	broken := func(_ context.Context, _ []int) ([]int, error) {
		return nil, AsDataError(errors.New("bulk response malformed"))
	}
	for _, test := range []struct {
		name      string
		mode      FailureMode
		eventType EventType
	}{
		{name: "quarantine", mode: Quarantine, eventType: EventQuarantined},
		{name: "skip", mode: Skip, eventType: EventSkipped},
	} {
		t.Run(test.name, func(t *testing.T) {
			base := bulkStep("bulk-" + test.name)
			base.Policy.OnError = test.mode
			ledger := &recordingLedger{}
			_, _, err := Run(t.Context(), Bulk(base, broken, 2), []int{1, 2}, Options{Ledger: ledger})
			require.NoError(t, err)
			events := ledger.byType(test.eventType)
			require.Len(t, events, 2)
			for index, event := range events {
				require.Equal(t, index, event.Index)
				require.Equal(t, DataError.String(), event.Class)
				require.Contains(t, event.Error, "bulk response malformed")
			}
		})
	}

	base := bulkStep("bulk-ledger-failure")
	base.Policy.OnError = Quarantine
	_, _, err := Run(t.Context(), Bulk(base, broken, 2), []int{1}, Options{Ledger: &recordingLedger{fail: true}})
	require.ErrorContains(t, err, "ledger event")
}

func TestBulkLengthMismatchIsFatal(t *testing.T) {
	step := Bulk(bulkStep("short-bulk"), func(_ context.Context, batch []int) ([]int, error) {
		return make([]int, len(batch)-1), nil
	}, 4)
	_, _, err := Run(context.Background(), step, []int{1, 2}, Options{Store: NewMemoryStore()})
	require.Error(t, err)
	require.Contains(t, err.Error(), "returned 1 results for 2 items")
}

func TestBulkOnResultSeesHitsAndFreshItems(t *testing.T) {
	store := NewMemoryStore()
	base := bulkStep("bulk-hook")
	var seen sync.Map
	base.OnResult = func(_ context.Context, index int, value int, outcome execution.CacheOutcome) error {
		seen.Store(index, outcome.State)
		return nil
	}
	step := Bulk(base, doubleAll(nil, nil, nil), 2)
	_, _, err := Run(context.Background(), step, []int{1, 2}, Options{Store: store})
	require.NoError(t, err)
	state, _ := seen.Load(0)
	require.Equal(t, execution.CacheStored, state)
	_, _, err = Run(context.Background(), step, []int{1, 2}, Options{Store: store})
	require.NoError(t, err)
	state, _ = seen.Load(1)
	require.Equal(t, execution.CacheHit, state)
}

func TestBulkInteroperatesWithMapCachedBatchesEntries(t *testing.T) {
	// The embeddings conversion must replay every vector the
	// MapCachedBatches era stored: same keys, same values.
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	require.NoError(t, err)
	type keyInput struct {
		Model         string `json:"model"`
		ItemID        string `json:"item_id"`
		ContentDigest string `json:"content_digest"`
	}
	items := []string{"alpha", "beta"}
	_, _, err = execution.MapCachedBatches(context.Background(), items,
		execution.CachedBatchMapOptions[string]{
			Workers: 1, BatchSize: 10, Cache: cache,
			Key: func(item string) (execution.Key, error) {
				return execution.NewKey("corpus-embedding", "v2", keyInput{Model: "m", ItemID: item, ContentDigest: "d-" + item})
			},
		},
		func(_ context.Context, batch []string) ([][]float32, error) {
			out := make([][]float32, len(batch))
			for index := range batch {
				out[index] = []float32{float32(index) + 0.5}
			}
			return out, nil
		},
	)
	require.NoError(t, err)

	base := Step[string, []float32]{
		Name: "corpus-embedding",
		Identity: Identity[string]{
			Kind:    "corpus-embedding",
			Version: "v2",
			Key: func(item string) ([]byte, error) {
				return []byte(`{"model":"m","item_id":"` + item + `","content_digest":"d-` + item + `"}`), nil
			},
		},
	}
	step := Bulk(base, func(_ context.Context, _ []string) ([][]float32, error) {
		return nil, errors.New("must not be called: every item is cached")
	}, 10)
	results, report, err := Run(context.Background(), step, items, Options{Store: cache})
	require.NoError(t, err)
	require.Equal(t, 2, report.Step("corpus-embedding").Hits)
	require.Equal(t, []float32{0.5}, results[0].Value)
	require.Equal(t, []float32{1.5}, results[1].Value)
}

func TestDeclareSharesBudgetsWithStepsAndReferences(t *testing.T) {
	shared := Options{}.Share()
	require.NoError(t, shared.Declare("harness", Resource{Name: "generation", Ceiling: 2, Budget: 2}))

	// A step referencing the declared resource by name draws from it.
	step := Step[int, int]{
		Name:   "consumer",
		Policy: Policy{Admission: []Resource{{Name: "generation"}}},
		Do:     func(_ context.Context, value int) (int, error) { return value, nil },
	}
	_, _, err := Run(context.Background(), step, []int{1, 2}, shared)
	require.NoError(t, err)
	_, _, err = Run(context.Background(), step, []int{3}, shared)
	require.ErrorIs(t, err, execution.ErrBudgetExceeded)

	require.NotNil(t, shared.Limiter("generation"))
	require.Nil(t, shared.Limiter("never-declared"))
	require.Equal(t, execution.BudgetSnapshot{Limit: 2, Spent: 2, Remaining: 0}, shared.Snapshots()["generation"])
}

func TestReferenceWithoutDeclarationBecomesZeroPlan(t *testing.T) {
	// A mistyped or undeclared reference must fail loudly on first spend —
	// a zero plan admits nothing — instead of silently minting budget.
	step := Step[int, int]{
		Name:   "orphan",
		Policy: Policy{Admission: []Resource{{Name: "ghost"}}},
		Do:     func(_ context.Context, value int) (int, error) { return value, nil },
	}
	_, _, err := Run(context.Background(), step, []int{1}, Options{})
	require.ErrorIs(t, err, execution.ErrBudgetExceeded)
	require.Contains(t, err.Error(), `resource "ghost"`)
}

func TestDeclareRunsPreflightAndReportsCost(t *testing.T) {
	unit := 0.5
	shared := Options{Preflight: &Preflight{MaxEstimatedUSD: 10}}.Share()
	require.NoError(t, shared.Declare("harness", Resource{Name: "priced", Ceiling: 4, Budget: 4, UnitUSD: &unit}))
	require.Equal(t, 2.0, shared.Cost().EstimatedUSD)

	err := shared.Declare("harness", Resource{Name: "too-expensive", Ceiling: 100, Budget: 100, UnitUSD: &unit})
	require.Error(t, err)
	require.Contains(t, err.Error(), "exceeds maximum USD")

	unshared := Options{}
	require.Error(t, unshared.Declare("x", Resource{Name: "r", Ceiling: 1, Budget: 1}))
}

func TestOnResultStreamsCompletionsAndErrorsFailTheRun(t *testing.T) {
	store := NewMemoryStore()
	var order []int
	var mutex sync.Mutex
	step := doubler("hooked", Policy{Workers: 1})
	step.OnResult = func(_ context.Context, index int, value int, outcome execution.CacheOutcome) error {
		mutex.Lock()
		order = append(order, index)
		mutex.Unlock()
		return nil
	}
	_, _, err := Run(context.Background(), step, []int{1, 2, 3}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, []int{0, 1, 2}, order)

	step.OnResult = func(context.Context, int, int, execution.CacheOutcome) error {
		return errors.New("append failed")
	}
	_, _, err = Run(context.Background(), step, []int{9}, Options{Store: store})
	require.Error(t, err)
	require.Contains(t, err.Error(), "result hook")
}

func TestOnResultSkipsQuarantinedItems(t *testing.T) {
	var hookCalls atomic.Int64
	step := Step[int, int]{
		Name:   "hook-quarantine",
		Policy: Policy{OnError: Quarantine},
		Do: func(_ context.Context, value int) (int, error) {
			if value == 1 {
				return 0, AsDataError(errors.New("bad"))
			}
			return value, nil
		},
		OnResult: func(context.Context, int, int, execution.CacheOutcome) error {
			hookCalls.Add(1)
			return nil
		},
	}
	_, _, err := Run(context.Background(), step, []int{0, 1, 2}, Options{})
	require.NoError(t, err)
	require.Equal(t, int64(2), hookCalls.Load())
}
