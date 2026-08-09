package generation

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/execution"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func generationRequest(text string) rag.GenerationRequest {
	return rag.GenerationRequest{
		Kind: "grounded-answer", Model: "chat-model", Prompt: "answer from evidence",
		Text: text, OutputSchema: `{"type":"object"}`,
		Evidence: []rag.Evidence{{Chunk: rag.Chunk{
			ID: "chunk-1", Text: "source text", ContentDigest: digest.Text("source text"),
		}}},
	}
}

func TestGenerationCacheKeySensitivity(t *testing.T) {
	base := generationRequest("question")
	baseKey, err := NewGenerationCacheKey(base, "adapter-v1", "evidence-only-v1")
	require.NoError(t, err)
	tests := map[string]func(*rag.GenerationRequest, *string, *string){
		"model":       func(r *rag.GenerationRequest, _, _ *string) { r.Model = "other" },
		"kind":        func(r *rag.GenerationRequest, _, _ *string) { r.Kind = "other" },
		"query":       func(r *rag.GenerationRequest, _, _ *string) { r.Text = "other" },
		"evidence id": func(r *rag.GenerationRequest, _, _ *string) { r.Evidence[0].Chunk.ID = "chunk-2" },
		"evidence digest": func(r *rag.GenerationRequest, _, _ *string) {
			r.Evidence[0].Chunk.Text = "changed source text"
			r.Evidence[0].Chunk.ContentDigest = digest.Text(r.Evidence[0].Chunk.Text)
		},
		"evidence order": func(r *rag.GenerationRequest, _, _ *string) {
			r.Evidence = append(r.Evidence, rag.Evidence{Chunk: rag.Chunk{ID: "chunk-2", Text: "second source", ContentDigest: digest.Text("second source")}})
			r.Evidence[0], r.Evidence[1] = r.Evidence[1], r.Evidence[0]
		},
		"prompt":          func(r *rag.GenerationRequest, _, _ *string) { r.Prompt = "other" },
		"schema":          func(r *rag.GenerationRequest, _, _ *string) { r.OutputSchema = "other" },
		"adapter version": func(_ *rag.GenerationRequest, a, _ *string) { *a = "adapter-v2" },
		"context policy":  func(_ *rag.GenerationRequest, _, p *string) { *p = "other" },
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			request := base
			request.Evidence = append([]rag.Evidence(nil), base.Evidence...)
			adapter, policy := "adapter-v1", "evidence-only-v1"
			mutate(&request, &adapter, &policy)
			key, err := NewGenerationCacheKey(request, adapter, policy)
			require.NoError(t, err)
			require.NotEqual(t, baseKey.InputDigest, key.InputDigest)
		})
	}
}

func TestGenerationCacheKeyIgnoresEvidenceScores(t *testing.T) {
	request := generationRequest("question")
	request.Evidence[0].RetrievalScore = 1.996638799554313
	first, err := NewGenerationCacheKey(request, "adapter-v1", "evidence-only-v1")
	require.NoError(t, err)

	rerankerScore := 0.75
	request.Evidence[0].RetrievalScore = 1.9966387995543127
	request.Evidence[0].RerankerScore = &rerankerScore
	request.Evidence[0].Rank = 99
	second, err := NewGenerationCacheKey(request, "adapter-v1", "evidence-only-v1")
	require.NoError(t, err)
	require.Equal(t, first, second)
}

type countingGenerator struct {
	calls  atomic.Int64
	failOn string
}

func (g *countingGenerator) Generate(_ context.Context, request rag.GenerationRequest) (rag.GenerationResult, error) {
	g.calls.Add(1)
	if request.Text == g.failOn {
		return rag.GenerationResult{}, errors.New("generation failed")
	}
	input, output, cost := int64(10), int64(2), 0.01
	return rag.GenerationResult{
		Text:  "answer:" + request.Text,
		Usage: rag.Usage{InputTokens: &input, OutputTokens: &output, CostUSD: &cost},
	}, nil
}

func cachedOptions(cache execution.Cache, limiter execution.Limiter) CachedProviderOptions {
	return CachedProviderOptions{
		Cache: cache, Limiter: limiter, Workers: 1,
		AdapterVersion: "adapter-v1", ContextPolicy: "evidence-only-v1",
	}
}

func TestGenerateCachedDuplicateMixedUsageAndZeroBudgetReplay(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	require.NoError(t, err)
	provider := &countingGenerator{}
	request := generationRequest("same")

	results, report, err := GenerateCached(t.Context(), []rag.GenerationRequest{request, request}, cachedOptions(cache, nil), provider)
	require.NoError(t, err)
	require.Len(t, results, 2)
	require.Equal(t, int64(1), provider.calls.Load())
	require.Equal(t, 2, report.Cache.Misses)
	require.Equal(t, 1, report.Cache.Writes)
	require.Equal(t, 1, report.Cache.WorkCalls)
	require.NotNil(t, report.Usage.CostUSD)
	require.Equal(t, 0.01, *report.Usage.CostUSD)
	require.Equal(t, rag.Usage{}, results[0].Usage)

	second := generationRequest("new")
	budget, err := execution.NewBudget(1)
	require.NoError(t, err)
	_, mixed, err := GenerateCached(t.Context(), []rag.GenerationRequest{request, second}, cachedOptions(cache, budget), provider)
	require.NoError(t, err)
	require.Equal(t, 1, mixed.Cache.Hits)
	require.Equal(t, 1, mixed.Cache.Misses)
	require.Equal(t, 1, budget.Spent())
	require.NotNil(t, mixed.Usage.InputTokens)
	require.Equal(t, int64(10), *mixed.Usage.InputTokens)

	zero, err := execution.NewBudget(0)
	require.NoError(t, err)
	_, replay, err := GenerateCached(t.Context(), []rag.GenerationRequest{request, second}, cachedOptions(cache, zero), provider)
	require.NoError(t, err)
	require.Equal(t, 2, replay.Cache.Hits)
	require.Equal(t, rag.Usage{}, replay.Usage)
	require.Zero(t, zero.Spent())
}

func TestGenerateCachedRecoversAfterLateFailure(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	require.NoError(t, err)
	requests := []rag.GenerationRequest{
		generationRequest("one"), generationRequest("two"), generationRequest("three"),
	}
	failing := &countingGenerator{failOn: "three"}
	_, first, err := GenerateCached(t.Context(), requests, cachedOptions(cache, nil), failing)
	require.Error(t, err)
	require.Equal(t, 2, first.Cache.Writes)
	require.Equal(t, 3, first.Cache.WorkCalls)

	budget, _ := execution.NewBudget(1)
	success := &countingGenerator{}
	_, recovered, err := GenerateCached(t.Context(), requests, cachedOptions(cache, budget), success)
	require.NoError(t, err)
	require.Equal(t, 2, recovered.Cache.Hits)
	require.Equal(t, int64(1), success.calls.Load())

	zero, _ := execution.NewBudget(0)
	_, replay, err := GenerateCached(t.Context(), requests, cachedOptions(cache, zero), &countingGenerator{failOn: "one"})
	require.NoError(t, err)
	require.Equal(t, 3, replay.Cache.Hits)
}

type failingStoreCache struct{}

func (failingStoreCache) Load(context.Context, execution.Key, any) (bool, error) {
	return false, nil
}
func (failingStoreCache) Store(context.Context, execution.Key, any) error {
	return errors.New("disk full")
}

func TestGenerateCachedReportsFailedWrite(t *testing.T) {
	_, report, err := GenerateCached(
		t.Context(), []rag.GenerationRequest{generationRequest("one")},
		cachedOptions(failingStoreCache{}, nil), &countingGenerator{},
	)
	require.ErrorContains(t, err, "store cache result")
	require.Zero(t, report.Cache.Writes)
	require.Equal(t, 1, report.Cache.WorkCalls)
}

func TestGenerateCachedFailsClosedOnCorruptEntry(t *testing.T) {
	root := t.TempDir()
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: root})
	require.NoError(t, err)
	request := generationRequest("one")
	_, _, err = GenerateCached(t.Context(), []rag.GenerationRequest{request}, cachedOptions(cache, nil), &countingGenerator{})
	require.NoError(t, err)
	var path string
	err = filepath.WalkDir(root, func(current string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !entry.IsDir() {
			path = current
		}
		return nil
	})
	require.NoError(t, err)
	require.NotEmpty(t, path)
	require.NoError(t, os.WriteFile(path, []byte(`{"corrupt":true}`), 0o600))
	provider := &countingGenerator{}
	_, _, err = GenerateCached(t.Context(), []rag.GenerationRequest{request}, cachedOptions(cache, nil), provider)
	require.ErrorIs(t, err, execution.ErrCorruptCache)
	require.Zero(t, provider.calls.Load())
}
