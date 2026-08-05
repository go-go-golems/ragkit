package execution

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"testing"
)

func TestMapCachedRecoversCompletedItemsAfterLateFailure(t *testing.T) {
	t.Parallel()

	cache, err := NewFileCache(FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatalf("NewFileCache() error = %v", err)
	}
	firstBudget, err := NewBudget(5)
	if err != nil {
		t.Fatalf("NewBudget(first) error = %v", err)
	}
	var firstCalls atomic.Int64
	key := func(value int) (Key, error) { return NewKey("embed", "v1", value) }
	_, firstReport, err := MapCached(
		context.Background(),
		[]int{1, 2, 3, 4, 5},
		CachedMapOptions[int]{
			Map:   MapOptions[int]{Workers: 1, Limiter: firstBudget},
			Cache: cache,
			Key:   key,
		},
		func(_ context.Context, value int) (int, error) {
			firstCalls.Add(1)
			if value == 5 {
				return 0, errors.New("provider failed")
			}
			return value * 10, nil
		},
	)
	if err == nil {
		t.Fatal("first MapCached() error = nil, want late failure")
	}
	if firstCalls.Load() != 5 || firstReport.WorkCalls != 5 || firstReport.Writes != 4 {
		t.Fatalf("first run calls=%d report=%+v", firstCalls.Load(), firstReport)
	}

	secondBudget, err := NewBudget(1)
	if err != nil {
		t.Fatalf("NewBudget(second) error = %v", err)
	}
	var secondCalls atomic.Int64
	results, secondReport, err := MapCached(
		context.Background(),
		[]int{1, 2, 3, 4, 5},
		CachedMapOptions[int]{
			Map:   MapOptions[int]{Workers: 2, Limiter: secondBudget},
			Cache: cache,
			Key:   key,
		},
		func(_ context.Context, value int) (int, error) {
			secondCalls.Add(1)
			return value * 10, nil
		},
	)
	if err != nil {
		t.Fatalf("second MapCached() error = %v", err)
	}
	if secondCalls.Load() != 1 {
		t.Fatalf("second run calls = %d, want 1", secondCalls.Load())
	}
	if secondBudget.Spent() != 1 {
		t.Fatalf("second budget spent = %d, want 1", secondBudget.Spent())
	}
	if secondReport.Hits != 4 || secondReport.Misses != 1 ||
		secondReport.Writes != 1 || secondReport.WorkCalls != 1 {
		t.Fatalf("second report = %+v", secondReport)
	}
	for index, result := range results {
		want := (index + 1) * 10
		if result != want {
			t.Fatalf("results[%d] = %d, want %d", index, result, want)
		}
	}
}

func TestMapCachedExecutesDuplicateKeysOnce(t *testing.T) {
	t.Parallel()

	cache, err := NewFileCache(FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatalf("NewFileCache() error = %v", err)
	}
	var calls atomic.Int64
	results, report, err := MapCached(
		context.Background(),
		[]string{"same", "same", "same"},
		CachedMapOptions[string]{
			Map:   MapOptions[string]{Workers: 3},
			Cache: cache,
			Key: func(value string) (Key, error) {
				return NewKey("step", "v1", value)
			},
		},
		func(_ context.Context, value string) (string, error) {
			calls.Add(1)
			return value + "-result", nil
		},
	)
	if err != nil {
		t.Fatalf("MapCached() error = %v", err)
	}
	if calls.Load() != 1 || report.WorkCalls != 1 || report.Writes != 1 || report.Misses != 3 {
		t.Fatalf("calls=%d report=%+v", calls.Load(), report)
	}
	for index, result := range results {
		if result != "same-result" {
			t.Fatalf("results[%d] = %q", index, result)
		}
	}
}

func TestMapCachedNotifiesEachDurableResultBeforeLateFailure(t *testing.T) {
	t.Parallel()

	cache, err := NewFileCache(FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	var mutex sync.Mutex
	notified := map[int]CacheState{}
	_, report, err := MapCached(
		context.Background(),
		[]int{1, 1, 2, 3},
		CachedMapOptions[int]{
			Map: MapOptions[int]{Workers: 1}, Cache: cache,
			Key: func(value int) (Key, error) { return NewKey("notify", "v1", value) },
		},
		func(_ context.Context, value int) (int, error) {
			if value == 3 {
				return 0, errors.New("late failure")
			}
			return value * 10, nil
		},
		func(index, value int, outcome CacheOutcome) error {
			mutex.Lock()
			defer mutex.Unlock()
			if value == 0 {
				t.Errorf("callback value = 0")
			}
			notified[index] = outcome.State
			return nil
		},
	)
	if err == nil {
		t.Fatal("MapCached() error = nil")
	}
	if report.Writes != 2 {
		t.Fatalf("writes = %d, want 2", report.Writes)
	}
	mutex.Lock()
	defer mutex.Unlock()
	if len(notified) != 3 {
		t.Fatalf("notified = %v, want indexes 0,1,2", notified)
	}
	for _, index := range []int{0, 1, 2} {
		if notified[index] != CacheStored {
			t.Fatalf("notified[%d] = %q, want stored", index, notified[index])
		}
	}
}

func TestMapCachedDoesNotRecomputeCorruptEntry(t *testing.T) {
	t.Parallel()

	cache, err := NewFileCache(FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatalf("NewFileCache() error = %v", err)
	}
	key, err := NewKey("embed", "v1", 1)
	if err != nil {
		t.Fatalf("NewKey() error = %v", err)
	}
	if err := cache.Store(context.Background(), key, 10); err != nil {
		t.Fatalf("Store() error = %v", err)
	}
	path, err := cache.path(key)
	if err != nil {
		t.Fatalf("path() error = %v", err)
	}
	if err := osWriteCorrupt(path); err != nil {
		t.Fatalf("osWriteCorrupt() error = %v", err)
	}

	var calls atomic.Int64
	_, report, err := MapCached(
		context.Background(),
		[]int{1},
		CachedMapOptions[int]{
			Cache: cache,
			Key:   func(int) (Key, error) { return key, nil },
		},
		func(_ context.Context, value int) (int, error) {
			calls.Add(1)
			return value * 10, nil
		},
	)
	if !errors.Is(err, ErrCorruptCache) {
		t.Fatalf("MapCached() error = %v, want ErrCorruptCache", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("work calls = %d, want 0", calls.Load())
	}
	if report.WorkCalls != 0 {
		t.Fatalf("reported work calls = %d, want 0", report.WorkCalls)
	}
}

func osWriteCorrupt(path string) error {
	return os.WriteFile(path, []byte(fmt.Sprintf(`{"bad":%q}`, path)), 0o600)
}
