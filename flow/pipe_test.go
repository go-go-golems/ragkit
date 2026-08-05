package flow

import (
	"context"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pkg/errors"
	"github.com/stretchr/testify/require"
)

func TestPipe2StreamsPerItem(t *testing.T) {
	// Stage one finishes item 0 immediately but sits on item 1; per-item
	// streaming means stage two must see item 0 while stage one still works.
	release := make(chan struct{})
	entered := make(chan int, 8)
	first := Step[int, int]{
		Name:   "slow-first",
		Policy: Policy{Workers: 2},
		Do: func(ctx context.Context, value int) (int, error) {
			if value == 1 {
				select {
				case <-release:
				case <-ctx.Done():
					return 0, ctx.Err()
				}
			}
			return value, nil
		},
	}
	second := Step[int, string]{
		Name:   "second",
		Policy: Policy{Workers: 2},
		Do: func(_ context.Context, value int) (string, error) {
			entered <- value
			if value == 0 {
				close(release) // item 0 reached stage two => release item 1
			}
			return strconv.Itoa(value), nil
		},
	}
	results, report, err := Run(context.Background(), Pipe2(first, second), []int{0, 1}, Options{})
	require.NoError(t, err)
	require.Equal(t, "0", results[0].Value)
	require.Equal(t, "1", results[1].Value)
	require.Equal(t, 0, <-entered, "item 0 must enter stage two before item 1 finished stage one")
	require.Equal(t, 2, report.Step("slow-first").Items)
	require.Equal(t, 2, report.Step("second").Items)
}

func TestPipe2PerStageReportsAndPolicies(t *testing.T) {
	store := NewMemoryStore()
	first := doubler("stage-a", Policy{Workers: 2})
	second := Step[int, string]{
		Name:   "stage-b",
		Policy: Policy{Workers: 1},
		Do: func(_ context.Context, value int) (string, error) {
			return strconv.Itoa(value), nil
		},
	}
	_, report, err := Run(context.Background(), Pipe2(first, second), []int{1, 2, 3}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, 3, report.Step("stage-a").Misses)
	require.Equal(t, 3, report.Step("stage-b").Items)
	require.Equal(t, 0, report.Step("stage-b").Misses, "uncached stage has no cache traffic")

	// Replay: stage-a hits, stage-b recomputes (uncached by design).
	_, report, err = Run(context.Background(), Pipe2(first, second), []int{1, 2, 3}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, 3, report.Step("stage-a").Hits)
	require.Equal(t, 3, report.Step("stage-b").WorkCalls)
}

func TestPipeQuarantinedItemsBypassLaterStages(t *testing.T) {
	var secondSaw []int
	var mutex sync.Mutex
	first := Step[int, int]{
		Name:   "quarantines",
		Policy: Policy{OnError: Quarantine},
		Do: func(_ context.Context, value int) (int, error) {
			if value == 1 {
				return 0, AsDataError(errors.New("bad item"))
			}
			return value, nil
		},
	}
	second := Step[int, int]{
		Name: "downstream",
		Do: func(_ context.Context, value int) (int, error) {
			mutex.Lock()
			secondSaw = append(secondSaw, value)
			mutex.Unlock()
			return value + 100, nil
		},
	}
	results, report, err := Run(context.Background(), Pipe2(first, second), []int{0, 1, 2}, Options{})
	require.NoError(t, err)
	require.Equal(t, 100, results[0].Value)
	require.Equal(t, 102, results[2].Value)
	require.NotNil(t, results[1].Quarantined)
	require.Equal(t, "quarantines", results[1].Quarantined.Step)
	require.Equal(t, 1, results[1].Quarantined.Index)
	mutex.Lock()
	require.NotContains(t, secondSaw, 1, "downstream must never see the quarantined item")
	mutex.Unlock()
	require.Equal(t, 2, report.Step("downstream").Items)
	require.Equal(t, 1, report.Step("quarantines").Quarantined)
}

func TestPipeBarrierWaitsForAllUpstreamResults(t *testing.T) {
	var maxSeen atomic.Int64
	var stageOneDone atomic.Int64
	first := Step[int, int]{
		Name:   "spread",
		Policy: Policy{Workers: 3},
		Do: func(_ context.Context, value int) (int, error) {
			time.Sleep(time.Duration(value) * 5 * time.Millisecond)
			stageOneDone.Add(1)
			return value, nil
		},
	}
	second := Step[int, int]{
		Name:    "barrier",
		Barrier: true,
		Policy:  Policy{Workers: 3},
		Do: func(_ context.Context, value int) (int, error) {
			done := stageOneDone.Load()
			if done > maxSeen.Load() {
				maxSeen.Store(done)
			}
			require.Equal(t, int64(4), done, "barrier stage must start after every upstream item")
			return value, nil
		},
	}
	_, _, err := Run(context.Background(), Pipe2(first, second), []int{0, 1, 2, 3}, Options{})
	require.NoError(t, err)
}

func TestPipe3FlattensAndOrdersResults(t *testing.T) {
	first := doubler("p3-first", Policy{Workers: 2})
	second := Step[int, int]{
		Name:   "p3-second",
		Policy: Policy{Workers: 2},
		Do:     func(_ context.Context, value int) (int, error) { return value + 1, nil },
	}
	third := Step[int, string]{
		Name:   "p3-third",
		Policy: Policy{Workers: 2},
		Do:     func(_ context.Context, value int) (string, error) { return strconv.Itoa(value), nil },
	}
	composed := Pipe3(first, second, third)
	require.Len(t, composed.stages, 3, "stages must flatten")

	nested := Pipe2(Pipe2(first, second), third)
	require.Len(t, nested.stages, 3, "nested pipes must flatten identically")

	items := []int{5, 1, 9, 3}
	results, report, err := Run(context.Background(), composed, items, Options{Store: NewMemoryStore()})
	require.NoError(t, err)
	for index, item := range items {
		require.Equal(t, strconv.Itoa(item*2+1), results[index].Value)
	}
	require.Equal(t, 4, report.Step("p3-first").Items)
	require.Equal(t, 4, report.Step("p3-second").Items)
	require.Equal(t, 4, report.Step("p3-third").Items)
}

func TestPipeStageFailureFailsRunWithStageName(t *testing.T) {
	first := doubler("ok-stage", Policy{})
	second := Step[int, int]{
		Name: "broken-stage",
		Do: func(_ context.Context, value int) (int, error) {
			return 0, errors.New("status=400: rejected")
		},
	}
	_, _, err := Run(context.Background(), Pipe2(first, second), []int{1}, Options{})
	require.Error(t, err)
	require.Contains(t, err.Error(), `stage "broken-stage"`)
	require.Contains(t, err.Error(), `step "broken-stage" item 0`)
}

func TestPipePreflightCoversEveryStageBeforeItemOne(t *testing.T) {
	var calls atomic.Int64
	first := Step[int, int]{
		Name:   "cheap",
		Policy: Policy{Admission: []Resource{{Name: "cheap-calls", Ceiling: 1, Budget: 1}}},
		Do: func(_ context.Context, value int) (int, error) {
			calls.Add(1)
			return value, nil
		},
	}
	second := Step[int, int]{
		Name:   "starved",
		Policy: Policy{Admission: []Resource{{Name: "starved-calls", Ceiling: 10, Budget: 1}}},
		Do: func(_ context.Context, value int) (int, error) {
			calls.Add(1)
			return value, nil
		},
	}
	_, _, err := Run(context.Background(), Pipe2(first, second), []int{1}, Options{
		Preflight: &Preflight{MaxEstimatedUSD: 1, AllowUnpriced: true},
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), `resource "starved-calls" budget 1 cannot cover the stated ceiling of 10`)
	require.Equal(t, int64(0), calls.Load(), "refusal must happen before any work")
}
