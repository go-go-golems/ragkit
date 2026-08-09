package execution

import (
	"context"
	"errors"
	"slices"
	"testing"
)

func TestMapCachedBatchesBoundsCallsAndPreservesOrder(t *testing.T) {
	t.Parallel()
	cache, err := NewFileCache(FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	budget, err := NewBudget(5)
	if err != nil {
		t.Fatal(err)
	}
	var calls [][]int
	results, report, err := MapCachedBatches(
		t.Context(),
		[]int{1, 2, 2, 3, 4, 5},
		CachedBatchMapOptions[int]{
			Workers: 1, Limiter: budget, BatchSize: 2, Cache: cache,
			Key: func(value int) (Key, error) { return NewKey("batch", "v1", value) },
		},
		func(_ context.Context, values []int) ([]int, error) {
			calls = append(calls, slices.Clone(values))
			outputs := make([]int, len(values))
			for index, value := range values {
				outputs[index] = value * 10
			}
			return outputs, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(results, []int{10, 20, 20, 30, 40, 50}) {
		t.Fatalf("results = %v", results)
	}
	if len(calls) != 3 || !slices.Equal(calls[0], []int{1, 2}) ||
		!slices.Equal(calls[1], []int{3, 4}) || !slices.Equal(calls[2], []int{5}) {
		t.Fatalf("calls = %v", calls)
	}
	if budget.Spent() != 5 || report.Misses != 6 ||
		report.Writes != 5 || report.WorkCalls != 3 {
		t.Fatalf("budget=%+v report=%+v", budget.Snapshot(), report)
	}
}

func TestMapCachedBatchesRecoversCompletedBatchesAndReplaysAtZeroBudget(t *testing.T) {
	t.Parallel()
	cache, err := NewFileCache(FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	key := func(value int) (Key, error) { return NewKey("recovery-batch", "v1", value) }
	firstBudget, _ := NewBudget(5)
	_, firstReport, err := MapCachedBatches(
		t.Context(),
		[]int{1, 2, 3, 4, 5},
		CachedBatchMapOptions[int]{
			Workers: 1, Limiter: firstBudget, BatchSize: 2, Cache: cache, Key: key,
		},
		func(_ context.Context, values []int) ([]int, error) {
			if slices.Contains(values, 5) {
				return nil, errors.New("late provider failure")
			}
			outputs := make([]int, len(values))
			for index, value := range values {
				outputs[index] = value * 10
			}
			return outputs, nil
		},
	)
	if err == nil {
		t.Fatal("first call error = nil")
	}
	if firstReport.Writes != 4 || firstReport.WorkCalls != 3 || firstBudget.Spent() != 5 {
		t.Fatalf("budget=%+v report=%+v", firstBudget.Snapshot(), firstReport)
	}

	secondBudget, _ := NewBudget(1)
	var secondCalls int
	results, secondReport, err := MapCachedBatches(
		t.Context(),
		[]int{1, 2, 3, 4, 5},
		CachedBatchMapOptions[int]{
			Workers: 1, Limiter: secondBudget, BatchSize: 2, Cache: cache, Key: key,
		},
		func(_ context.Context, values []int) ([]int, error) {
			secondCalls++
			return []int{values[0] * 10}, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if secondCalls != 1 || secondBudget.Spent() != 1 ||
		secondReport.Hits != 4 || secondReport.Writes != 1 || secondReport.WorkCalls != 1 {
		t.Fatalf("calls=%d budget=%+v report=%+v", secondCalls, secondBudget.Snapshot(), secondReport)
	}
	if !slices.Equal(results, []int{10, 20, 30, 40, 50}) {
		t.Fatalf("results = %v", results)
	}

	zeroBudget, _ := NewBudget(0)
	replayCalls := 0
	_, replayReport, err := MapCachedBatches(
		t.Context(),
		[]int{1, 2, 3, 4, 5},
		CachedBatchMapOptions[int]{
			Workers: 1, Limiter: zeroBudget, BatchSize: 2, Cache: cache, Key: key,
		},
		func(context.Context, []int) ([]int, error) {
			replayCalls++
			return nil, errors.New("must not execute")
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if replayCalls != 0 || zeroBudget.Spent() != 0 ||
		replayReport.Hits != 5 || replayReport.WorkCalls != 0 {
		t.Fatalf("calls=%d budget=%+v report=%+v", replayCalls, zeroBudget.Snapshot(), replayReport)
	}
}

func TestMapCachedBatchesRejectsWrongResultCountWithoutCaching(t *testing.T) {
	t.Parallel()
	cache, err := NewFileCache(FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	_, report, err := MapCachedBatches(
		t.Context(),
		[]int{1, 2},
		CachedBatchMapOptions[int]{
			BatchSize: 2, Cache: cache,
			Key: func(value int) (Key, error) { return NewKey("count", "v1", value) },
		},
		func(context.Context, []int) ([]int, error) { return []int{10}, nil },
	)
	if err == nil {
		t.Fatal("error = nil")
	}
	if report.Writes != 0 || report.WorkCalls != 1 {
		t.Fatalf("report = %+v", report)
	}
}
