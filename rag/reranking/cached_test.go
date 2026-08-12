package reranking

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/go-go-golems/flowkit/execution"
	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func rerankingRequest(query string) rag.RerankRequest {
	return rag.RerankRequest{
		Model: "rerank-model", Query: rag.Query{Text: query}, Results: 1,
		Candidates: []rag.Evidence{{Chunk: rag.Chunk{
			ID: "chunk-1", Text: "source text", ContentDigest: digest.Text("source text"),
		}}},
	}
}

func TestCacheKeySensitivity(t *testing.T) {
	base := rerankingRequest("question")
	baseKey, err := NewCacheKey(base, "adapter-v1")
	require.NoError(t, err)
	tests := map[string]func(*rag.RerankRequest, *string){
		"model":        func(r *rag.RerankRequest, _ *string) { r.Model = "other" },
		"query":        func(r *rag.RerankRequest, _ *string) { r.Query.Text = "other" },
		"candidate id": func(r *rag.RerankRequest, _ *string) { r.Candidates[0].Chunk.ID = "chunk-2" },
		"candidate digest": func(r *rag.RerankRequest, _ *string) {
			r.Candidates[0].Chunk.Text = "changed source text"
			r.Candidates[0].Chunk.ContentDigest = digest.Text(r.Candidates[0].Chunk.Text)
		},
		"candidate order": func(r *rag.RerankRequest, _ *string) {
			r.Candidates = append(r.Candidates, rag.Evidence{Chunk: rag.Chunk{
				ID: "chunk-2", Text: "second source", ContentDigest: digest.Text("second source"),
			}})
			r.Candidates[0], r.Candidates[1] = r.Candidates[1], r.Candidates[0]
		},
		"result count":    func(r *rag.RerankRequest, _ *string) { r.Results = 2 },
		"adapter version": func(_ *rag.RerankRequest, adapter *string) { *adapter = "adapter-v2" },
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			request := base
			request.Candidates = append([]rag.Evidence(nil), base.Candidates...)
			adapter := "adapter-v1"
			mutate(&request, &adapter)
			key, err := NewCacheKey(request, adapter)
			require.NoError(t, err)
			require.NotEqual(t, baseKey.InputDigest, key.InputDigest)
		})
	}
}

func TestCacheKeyIgnoresCandidateScores(t *testing.T) {
	request := rerankingRequest("question")
	request.Candidates[0].RetrievalScore = 1.996638799554313
	first, err := NewCacheKey(request, "adapter-v1")
	require.NoError(t, err)

	rerankerScore := 0.75
	request.Candidates[0].RetrievalScore = 1.9966387995543127
	request.Candidates[0].RerankerScore = &rerankerScore
	request.Candidates[0].Rank = 99
	second, err := NewCacheKey(request, "adapter-v1")
	require.NoError(t, err)
	require.Equal(t, first, second)
}

type countingReranker struct {
	calls  atomic.Int64
	failOn string
}

func (r *countingReranker) Rerank(
	_ context.Context,
	request rag.RerankRequest,
) (rag.RerankResult, error) {
	r.calls.Add(1)
	if request.Query.Text == r.failOn {
		return rag.RerankResult{}, errors.New("simulated reranking failure")
	}
	input, cost := int64(4), 0.005
	return rag.RerankResult{
		Evidence: request.Candidates,
		Usage:    rag.Usage{InputTokens: &input, CostUSD: &cost},
	}, nil
}

func cachedOptions(cache execution.Cache, limiter execution.Limiter) CachedOptions {
	return CachedOptions{
		Cache: cache, Limiter: limiter, Workers: 1, AdapterVersion: "adapter-v1",
	}
}

func TestCachedDeduplicatesAndReplays(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	require.NoError(t, err)
	request := rerankingRequest("question")
	provider := &countingReranker{}
	options := cachedOptions(cache, nil)
	results, first, err := Cached(
		t.Context(), []rag.RerankRequest{request, request}, options, provider,
	)
	require.NoError(t, err)
	require.Len(t, results, 2)
	require.Equal(t, int64(1), provider.calls.Load())
	require.Equal(t, 1, first.Cache.Writes)
	require.NotNil(t, first.Usage.CostUSD)

	zero, err := execution.NewBudget(0)
	require.NoError(t, err)
	options.Limiter = zero
	_, replay, err := Cached(t.Context(), []rag.RerankRequest{request}, options, provider)
	require.NoError(t, err)
	require.Equal(t, 1, replay.Cache.Hits)
	require.Equal(t, int64(1), provider.calls.Load())
	require.Equal(t, rag.Usage{}, replay.Usage)
}

func TestCachedRecoversAfterFinalItemFailure(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	require.NoError(t, err)
	requests := []rag.RerankRequest{
		rerankingRequest("one"), rerankingRequest("two"), rerankingRequest("three"),
	}
	options := cachedOptions(cache, nil)
	failing := &countingReranker{failOn: "three"}
	_, first, err := Cached(t.Context(), requests, options, failing)
	require.Error(t, err)
	require.Equal(t, 2, first.Cache.Writes)

	budget, err := execution.NewBudget(1)
	require.NoError(t, err)
	options.Limiter = budget
	success := &countingReranker{}
	_, recovered, err := Cached(t.Context(), requests, options, success)
	require.NoError(t, err)
	require.Equal(t, 2, recovered.Cache.Hits)
	require.Equal(t, int64(1), success.calls.Load())
	require.Equal(t, 1, budget.Spent())

	zero, err := execution.NewBudget(0)
	require.NoError(t, err)
	options.Limiter = zero
	replay := &countingReranker{failOn: "one"}
	_, replayReport, err := Cached(t.Context(), requests, options, replay)
	require.NoError(t, err)
	require.Equal(t, 3, replayReport.Cache.Hits)
	require.Zero(t, replay.calls.Load())
	require.Zero(t, zero.Spent())
}

type failedUsageReranker struct{}

func (failedUsageReranker) Rerank(context.Context, rag.RerankRequest) (rag.RerankResult, error) {
	tokens := int64(8)
	return rag.RerankResult{Usage: rag.Usage{InputTokens: &tokens}}, errors.New("provider failed")
}

func TestCachedReportsUsageFromFailedProviderCall(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	require.NoError(t, err)
	_, report, err := Cached(t.Context(), []rag.RerankRequest{rerankingRequest("one")}, cachedOptions(cache, nil), failedUsageReranker{})
	require.Error(t, err)
	require.NotNil(t, report.Usage.InputTokens)
	require.Equal(t, int64(8), *report.Usage.InputTokens)
}

func TestCachedRerankerReturnsKnownUsageWithError(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	require.NoError(t, err)
	decorator, err := NewCachedReranker(failedUsageReranker{}, cachedOptions(cache, nil))
	require.NoError(t, err)
	result, err := decorator.Rerank(t.Context(), rerankingRequest("one"))
	require.ErrorContains(t, err, "provider failed")
	require.NotNil(t, result.Usage.InputTokens)
	require.EqualValues(t, 8, *result.Usage.InputTokens)
}

func TestCachedRerankerDecoratesSingleRequestInterface(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	require.NoError(t, err)
	provider := &countingReranker{}
	decorator, err := NewCachedReranker(provider, cachedOptions(cache, nil))
	require.NoError(t, err)
	request := rerankingRequest("question")

	first, err := decorator.Rerank(t.Context(), request)
	require.NoError(t, err)
	require.Empty(t, first.Usage)
	second, err := decorator.Rerank(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, first.Evidence, second.Evidence)
	require.Equal(t, int64(1), provider.calls.Load())

	report := decorator.Snapshot()
	require.Equal(t, 1, report.Cache.Misses)
	require.Equal(t, 1, report.Cache.Hits)
	require.Equal(t, 1, report.Cache.Writes)
	require.Equal(t, 1, report.Cache.WorkCalls)
	require.Len(t, report.Cache.Outcomes, 2)
	require.NotNil(t, report.Usage.CostUSD)
	originalCost := *report.Usage.CostUSD

	report.Cache.Outcomes[0].KeyDigest = "changed"
	*report.Usage.CostUSD = 999
	require.NotEqual(t, report.Cache.Outcomes[0].KeyDigest, decorator.Snapshot().Cache.Outcomes[0].KeyDigest)
	require.Equal(t, originalCost, *decorator.Snapshot().Usage.CostUSD)
}
