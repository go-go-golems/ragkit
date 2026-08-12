package embedding

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"

	"github.com/go-go-golems/flowkit/execution"
	"github.com/go-go-golems/ragkit/rag"
)

type failingTextEmbedder struct {
	failOn string
	calls  atomic.Int64
}

type billedFailingEmbedder struct{}

func (billedFailingEmbedder) Embed(context.Context, rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	tokens := int64(7)
	cost := 0.25
	return rag.EmbeddingResult{Usage: rag.Usage{EmbeddingTokens: &tokens, CostUSD: &cost}}, errors.New("billed failure")
}

func (e *failingTextEmbedder) Embed(_ context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	e.calls.Add(1)
	if len(request.Texts) == 1 && request.Texts[0] == e.failOn {
		return rag.EmbeddingResult{}, errors.New("simulated embedding failure")
	}
	return rag.EmbeddingResult{Vectors: [][]float32{{float32(len(request.Texts[0]))}}}, nil
}

type countingEmbedder struct {
	calls atomic.Int64
}

type vectorByTextEmbedder struct {
	calls  atomic.Int64
	vector func(string) []float32
}

func (e *vectorByTextEmbedder) Embed(_ context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	e.calls.Add(1)
	return rag.EmbeddingResult{Vectors: [][]float32{e.vector(request.Texts[0])}}, nil
}

func (e *countingEmbedder) Embed(_ context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	e.calls.Add(1)
	vectors := make([][]float32, len(request.Texts))
	for index, text := range request.Texts {
		vectors[index] = []float32{float32(len(text))}
	}
	tokens := int64(2 * len(request.Texts))
	cost := 0.01 * float64(len(request.Texts))
	return rag.EmbeddingResult{
		Vectors: vectors,
		Usage:   rag.Usage{EmbeddingTokens: &tokens, CostUSD: &cost},
	}, nil
}

func TestCachedEmbedderCachesBeforeBudgetAdmission(t *testing.T) {
	t.Parallel()
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	provider := &countingEmbedder{}
	firstBudget, _ := execution.NewBudget(2)
	first, err := NewCachedEmbedder(provider, CachedEmbedderOptions{
		Cache: cache, Limiter: firstBudget, Workers: 2, Step: "query-embedding",
	})
	if err != nil {
		t.Fatal(err)
	}
	request := rag.EmbeddingRequest{Model: "model", Texts: []string{"oak", "maple"}}
	firstResult, err := first.Embed(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if firstResult.Usage.EmbeddingTokens == nil || *firstResult.Usage.EmbeddingTokens != 4 {
		t.Fatalf("first embedding tokens = %v, want 4", firstResult.Usage.EmbeddingTokens)
	}
	if firstResult.Usage.CostUSD == nil || *firstResult.Usage.CostUSD != 0.02 {
		t.Fatalf("first cost = %v, want 0.02", firstResult.Usage.CostUSD)
	}
	secondBudget, _ := execution.NewBudget(0)
	second, err := NewCachedEmbedder(provider, CachedEmbedderOptions{
		Cache: cache, Limiter: secondBudget, Workers: 2, Step: "query-embedding",
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := second.Embed(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Vectors) != 2 || provider.calls.Load() != 2 || secondBudget.Spent() != 0 {
		t.Fatalf("vectors=%d calls=%d spent=%d", len(result.Vectors), provider.calls.Load(), secondBudget.Spent())
	}
	if result.Usage != (rag.Usage{}) {
		t.Fatalf("cached usage = %+v, want no charged usage", result.Usage)
	}
	if got := second.Snapshot(); got.Hits != 2 || got.Writes != 0 || got.WorkCalls != 0 {
		t.Fatalf("snapshot = %+v", got)
	}
}

func TestCachedEmbedderReportsOnlyMissUsage(t *testing.T) {
	t.Parallel()
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	provider := &countingEmbedder{}
	embedder, err := NewCachedEmbedder(provider, CachedEmbedderOptions{
		Cache: cache, Workers: 2, Step: "query-embedding",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := embedder.Embed(t.Context(), rag.EmbeddingRequest{
		Model: "model", Texts: []string{"oak"},
	}); err != nil {
		t.Fatal(err)
	}
	result, err := embedder.Embed(t.Context(), rag.EmbeddingRequest{
		Model: "model", Texts: []string{"oak", "maple"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.EmbeddingTokens == nil || *result.Usage.EmbeddingTokens != 2 {
		t.Fatalf("embedding tokens = %v, want usage from one miss", result.Usage.EmbeddingTokens)
	}
	if result.Usage.CostUSD == nil || *result.Usage.CostUSD != 0.01 {
		t.Fatalf("cost = %v, want usage from one miss", result.Usage.CostUSD)
	}
}

func TestCachedEmbedderBudgetsQueryMisses(t *testing.T) {
	t.Parallel()
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	provider := &countingEmbedder{}
	budget, _ := execution.NewBudget(0)
	embedder, err := NewCachedEmbedder(provider, CachedEmbedderOptions{
		Cache: cache, Limiter: budget, Step: "query-embedding",
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = embedder.Embed(context.Background(), rag.EmbeddingRequest{Model: "model", Texts: []string{"oak"}})
	if err == nil {
		t.Fatal("Embed() error = nil, want budget rejection")
	}
	if provider.calls.Load() != 0 {
		t.Fatalf("provider calls = %d, want 0", provider.calls.Load())
	}
}

func TestCachedEmbedderRecordsUsageReturnedWithProviderError(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	embedder, err := NewCachedEmbedder(billedFailingEmbedder{}, CachedEmbedderOptions{Cache: cache})
	if err != nil {
		t.Fatal(err)
	}
	result, err := embedder.Embed(t.Context(), rag.EmbeddingRequest{Model: "model", Texts: []string{"oak"}})
	if err == nil {
		t.Fatal("Embed() error = nil, want provider failure")
	}
	usage := embedder.Snapshot().Usage
	if usage.EmbeddingTokens == nil || *usage.EmbeddingTokens != 7 || usage.CostUSD == nil || *usage.CostUSD != 0.25 {
		t.Fatalf("snapshot usage = %+v", usage)
	}
	if result.Usage.EmbeddingTokens == nil || *result.Usage.EmbeddingTokens != 7 || result.Usage.CostUSD == nil || *result.Usage.CostUSD != 0.25 {
		t.Fatalf("returned usage = %+v", result.Usage)
	}
}

func TestCachedEmbedderRecoversAfterFinalItemFailure(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	failing := &failingTextEmbedder{failOn: "cedar"}
	first, err := NewCachedEmbedder(failing, CachedEmbedderOptions{
		Cache: cache, Workers: 1, Step: "recovery-embedding",
	})
	if err != nil {
		t.Fatal(err)
	}
	request := rag.EmbeddingRequest{Model: "model", Texts: []string{"oak", "maple", "cedar"}}
	if _, err := first.Embed(t.Context(), request); err == nil {
		t.Fatal("first Embed() error = nil")
	}
	if snapshot := first.Snapshot(); snapshot.Writes != 2 || snapshot.WorkCalls != 3 {
		t.Fatalf("first writes = %d, want 2", snapshot.Writes)
	}

	budget, _ := execution.NewBudget(1)
	success := &failingTextEmbedder{}
	second, err := NewCachedEmbedder(success, CachedEmbedderOptions{
		Cache: cache, Limiter: budget, Workers: 1, Step: "recovery-embedding",
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := second.Embed(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Vectors) != 3 || success.calls.Load() != 1 ||
		second.Snapshot().WorkCalls != 1 || budget.Spent() != 1 {
		t.Fatalf("vectors=%d calls=%d spent=%d", len(result.Vectors), success.calls.Load(), budget.Spent())
	}

	zero, _ := execution.NewBudget(0)
	replayProvider := &failingTextEmbedder{failOn: "oak"}
	replay, err := NewCachedEmbedder(replayProvider, CachedEmbedderOptions{
		Cache: cache, Limiter: zero, Workers: 1, Step: "recovery-embedding",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := replay.Embed(t.Context(), request); err != nil {
		t.Fatal(err)
	}
	if replayProvider.calls.Load() != 0 || replay.Snapshot().WorkCalls != 0 || zero.Spent() != 0 {
		t.Fatalf("replay calls=%d spent=%d", replayProvider.calls.Load(), zero.Spent())
	}
}

func TestCachedEmbedderDoesNotCacheMalformedProviderVectors(t *testing.T) {
	for name, vector := range map[string][]float32{
		"empty": {}, "nan": {float32(math.NaN())}, "infinite": {float32(math.Inf(1))},
	} {
		t.Run(name, func(t *testing.T) {
			cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
			if err != nil {
				t.Fatal(err)
			}
			bad := &vectorByTextEmbedder{vector: func(string) []float32 { return vector }}
			first, err := NewCachedEmbedder(bad, CachedEmbedderOptions{Cache: cache, Step: "validated-embedding"})
			if err != nil {
				t.Fatal(err)
			}
			request := rag.EmbeddingRequest{Model: "model", Texts: []string{"oak"}}
			if _, err := first.Embed(t.Context(), request); err == nil {
				t.Fatal("Embed() error = nil")
			}

			recovered := &vectorByTextEmbedder{vector: func(string) []float32 { return []float32{1} }}
			second, err := NewCachedEmbedder(recovered, CachedEmbedderOptions{Cache: cache, Step: "validated-embedding"})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := second.Embed(t.Context(), request); err != nil {
				t.Fatal(err)
			}
			if recovered.calls.Load() != 1 {
				t.Fatalf("recovered provider calls = %d, malformed entry was cached", recovered.calls.Load())
			}
		})
	}
}

func TestCachedEmbedderRejectsCrossTextDimensionChangesBeforeCachingMismatch(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	changing := &vectorByTextEmbedder{vector: func(text string) []float32 {
		if text == "maple" {
			return []float32{1, 2}
		}
		return []float32{1}
	}}
	first, err := NewCachedEmbedder(changing, CachedEmbedderOptions{Cache: cache, Workers: 1, Step: "dimensioned-embedding"})
	if err != nil {
		t.Fatal(err)
	}
	request := rag.EmbeddingRequest{Model: "model", Texts: []string{"oak", "maple"}}
	if _, err := first.Embed(t.Context(), request); err == nil {
		t.Fatal("Embed() error = nil")
	}

	recovered := &vectorByTextEmbedder{vector: func(string) []float32 { return []float32{1} }}
	second, err := NewCachedEmbedder(recovered, CachedEmbedderOptions{Cache: cache, Workers: 1, Step: "dimensioned-embedding"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := second.Embed(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	if recovered.calls.Load() != 1 || len(result.Vectors[0]) != 1 || len(result.Vectors[1]) != 1 {
		t.Fatalf("calls=%d vectors=%v", recovered.calls.Load(), result.Vectors)
	}
}
