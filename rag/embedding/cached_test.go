package embedding

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-go-golems/ragkit/execution"
	"github.com/go-go-golems/ragkit/flow"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func TestCachedPreservesOrderDeduplicatesAndAccountsForMisses(t *testing.T) {
	t.Parallel()
	cache := newTestCache(t)
	embedder := &recordingEmbedder{}
	items := []Item{
		{ID: "b", Text: "two", ContentDigest: "digest-b"},
		{ID: "a", Text: "one", ContentDigest: "digest-a"},
		{ID: "b", Text: "two", ContentDigest: "digest-b"},
	}
	result, err := Cached(context.Background(), embedder, items, cachedTestOptions(cache, nil))
	require.NoError(t, err)
	require.Equal(t, []string{"b", "a", "b"}, vectorIDs(result.Vectors))
	require.Equal(t, 2, result.Cache.Writes)
	require.Equal(t, 3, result.Cache.Misses)
	require.Equal(t, 1, result.Cache.WorkCalls)
	require.Equal(t, 2, embedder.textCount())
	require.NotNil(t, result.Usage.EmbeddingTokens)
	require.EqualValues(t, 2, *result.Usage.EmbeddingTokens)

	replay, err := Cached(context.Background(), embedder, items, cachedTestOptions(cache, nil))
	require.NoError(t, err)
	require.Equal(t, 3, replay.Cache.Hits)
	require.Equal(t, 0, replay.Cache.WorkCalls)
	require.Equal(t, rag.Usage{}, replay.Usage)
	require.Equal(t, 1, embedder.callCount())
}

func TestCachedRecoversCompletedBatchesAndReplaysAtZeroBudget(t *testing.T) {
	t.Parallel()
	cache := newTestCache(t)
	items := testItems(5)
	failing := &recordingEmbedder{failText: "text-4"}
	firstOptions, firstFlow := budgetedTestOptions(cache, 5)
	firstOptions.BatchSize = 2
	first, err := Cached(context.Background(), failing, items, firstOptions)
	require.Error(t, err)
	require.Equal(t, 4, first.Cache.Writes)
	require.Equal(t, 3, first.Cache.WorkCalls)
	require.Equal(t, 5, firstFlow.Snapshots()["embedding"].Spent)

	secondOptions, secondFlow := budgetedTestOptions(cache, 1)
	secondOptions.BatchSize = 2
	secondEmbedder := &recordingEmbedder{}
	second, err := Cached(context.Background(), secondEmbedder, items, secondOptions)
	require.NoError(t, err)
	require.Equal(t, 4, second.Cache.Hits)
	require.Equal(t, 1, second.Cache.Misses)
	require.Equal(t, 1, second.Cache.Writes)
	require.Equal(t, 1, second.Cache.WorkCalls)
	require.Equal(t, 1, secondFlow.Snapshots()["embedding"].Spent)

	replayOptions, replayFlow := budgetedTestOptions(cache, 0)
	replayEmbedder := &recordingEmbedder{}
	replay, err := Cached(context.Background(), replayEmbedder, items, replayOptions)
	require.NoError(t, err)
	require.Equal(t, 5, replay.Cache.Hits)
	require.Equal(t, 0, replay.Cache.WorkCalls)
	require.Equal(t, 0, replayFlow.Snapshots()["embedding"].Spent)
	require.Equal(t, 0, replayEmbedder.callCount())
}

// TestCachedRetriesTransientBatchFailures covers the coverage gap that
// killed a complete 13,847-call build at embeddings: transient provider
// errors now retry under the cache.
func TestCachedRetriesTransientBatchFailures(t *testing.T) {
	t.Parallel()
	cache := newTestCache(t)
	var calls atomic.Int64
	flaky := embeddingFunc(func(_ context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
		if calls.Add(1) == 1 {
			return rag.EmbeddingResult{}, fmt.Errorf("read tcp: unexpected EOF")
		}
		vectors := make([][]float32, len(request.Texts))
		for index := range vectors {
			vectors[index] = []float32{1}
		}
		return rag.EmbeddingResult{Vectors: vectors}, nil
	})
	options := cachedTestOptions(cache, nil)
	options.Retry = flow.RetrySpec{Attempts: 3, Backoff: flow.Backoff{Base: time.Millisecond, Cap: time.Millisecond}}
	result, err := Cached(context.Background(), flaky, testItems(2), options)
	require.NoError(t, err)
	require.Equal(t, int64(2), calls.Load())
	require.Equal(t, 2, result.Cache.Writes)
}

func TestCachedRejectsInvalidProviderVectors(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		embedder rag.Embedder
	}{
		{
			name: "wrong count",
			embedder: embeddingFunc(func(_ context.Context, _ rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
				return rag.EmbeddingResult{}, nil
			}),
		},
		{
			name: "non finite",
			embedder: embeddingFunc(func(_ context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
				vectors := make([][]float32, len(request.Texts))
				for index := range vectors {
					vectors[index] = []float32{float32(math.NaN())}
				}
				return rag.EmbeddingResult{Vectors: vectors}, nil
			}),
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := Cached(
				context.Background(),
				test.embedder,
				testItems(2),
				cachedTestOptions(newTestCache(t), nil),
			)
			require.Error(t, err)
		})
	}
}

func TestCachedReportsUsageWhenProviderVectorsAreInvalid(t *testing.T) {
	t.Parallel()
	tokens := int64(3)
	embedder := embeddingFunc(func(_ context.Context, _ rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
		return rag.EmbeddingResult{
			Vectors: nil,
			Usage:   rag.Usage{EmbeddingTokens: &tokens},
		}, nil
	})
	result, err := Cached(context.Background(), embedder, testItems(1), cachedTestOptions(newTestCache(t), nil))
	require.Error(t, err)
	require.NotNil(t, result.Usage.EmbeddingTokens)
	require.EqualValues(t, 3, *result.Usage.EmbeddingTokens)
}

func TestCachedRejectsDimensionsThatDifferAcrossBatches(t *testing.T) {
	t.Parallel()
	cache := newTestCache(t)
	embedder := &varyingDimensionEmbedder{}
	options := cachedTestOptions(cache, nil)
	options.Workers = 1
	options.BatchSize = 1
	result, err := Cached(context.Background(), embedder, testItems(2), options)
	require.ErrorContains(t, err, "dimensions differ across batches")
	require.Equal(t, 1, result.Cache.Writes)
}

type embeddingFunc func(context.Context, rag.EmbeddingRequest) (rag.EmbeddingResult, error)

func (function embeddingFunc) Embed(ctx context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	return function(ctx, request)
}

type recordingEmbedder struct {
	mutex    sync.Mutex
	calls    int
	texts    int
	failText string
}

func (embedder *recordingEmbedder) Embed(_ context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	embedder.mutex.Lock()
	defer embedder.mutex.Unlock()
	embedder.calls++
	embedder.texts += len(request.Texts)
	vectors := make([][]float32, len(request.Texts))
	for index, text := range request.Texts {
		if text == embedder.failText {
			return rag.EmbeddingResult{}, fmt.Errorf("forced embedding failure")
		}
		vectors[index] = []float32{float32(len(text)), float32(index + 1)}
	}
	tokens := int64(len(request.Texts))
	return rag.EmbeddingResult{
		Vectors: vectors,
		Usage:   rag.Usage{EmbeddingTokens: &tokens},
	}, nil
}

func (embedder *recordingEmbedder) callCount() int {
	embedder.mutex.Lock()
	defer embedder.mutex.Unlock()
	return embedder.calls
}

func (embedder *recordingEmbedder) textCount() int {
	embedder.mutex.Lock()
	defer embedder.mutex.Unlock()
	return embedder.texts
}

type varyingDimensionEmbedder struct {
	calls int
}

func (embedder *varyingDimensionEmbedder) Embed(_ context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	embedder.calls++
	values := []float32{1}
	if embedder.calls > 1 {
		values = []float32{1, 2}
	}
	return rag.EmbeddingResult{Vectors: [][]float32{values}}, nil
}

func newTestCache(t *testing.T) execution.Cache {
	t.Helper()
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	require.NoError(t, err)
	return cache
}

func cachedTestOptions(cache execution.Cache, _ execution.Limiter) CachedOptions {
	return CachedOptions{
		Model: "model", Workers: 1, BatchSize: 8,
		Step: "test-embedding", Version: "v1",
		Flow: flow.Options{Store: cache},
	}
}

// budgetedTestOptions declares one "embedding" plan on shared flow options
// so tests can assert spend through snapshots.
func budgetedTestOptions(cache execution.Cache, budget int) (CachedOptions, flow.Options) {
	shared := flow.Options{Store: cache}.Share()
	options := CachedOptions{
		Model: "model", Workers: 1, BatchSize: 8,
		Step: "test-embedding", Version: "v1",
		Resource: flow.Resource{Name: "embedding", Ceiling: budget, Budget: budget},
		Flow:     shared,
	}
	return options, shared
}

func testItems(count int) []Item {
	items := make([]Item, count)
	for index := range items {
		items[index] = Item{
			ID:            fmt.Sprintf("item-%d", index),
			Text:          fmt.Sprintf("text-%d", index),
			ContentDigest: fmt.Sprintf("digest-%d", index),
		}
	}
	return items
}

func vectorIDs(vectors []rag.Vector) []string {
	ids := make([]string, len(vectors))
	for index, current := range vectors {
		ids[index] = current.RepresentationID
	}
	return ids
}
