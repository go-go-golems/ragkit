package generation

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/go-go-golems/ragkit/execution"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

type cachedGeneratorCountingProvider struct{ calls atomic.Int64 }

func (g *cachedGeneratorCountingProvider) Generate(
	_ context.Context,
	_ rag.GenerationRequest,
) (rag.GenerationResult, error) {
	g.calls.Add(1)
	tokens := int64(3)
	return rag.GenerationResult{
		Text: "answer", Usage: rag.Usage{OutputTokens: &tokens},
	}, nil
}

func TestCachedGeneratorRecoversBeforeZeroBudgetAdmission(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{
		Directory: t.TempDir(),
	})
	require.NoError(t, err)
	firstBudget, err := execution.NewBudget(1)
	require.NoError(t, err)
	provider := &cachedGeneratorCountingProvider{}
	first, err := NewCachedGenerator(provider, CachedProviderOptions{
		Cache: cache, Limiter: firstBudget, Workers: 1,
		AdapterVersion: "adapter-v1",
		ContextPolicy:  ContextPolicyNameForTest,
	})
	require.NoError(t, err)
	request := rag.GenerationRequest{
		Kind: "answer", Model: "model", Prompt: "prompt", Text: "question",
	}
	_, err = first.Generate(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, int64(1), provider.calls.Load())
	require.Equal(t, execution.CacheStored, first.LastOutcome().State)

	zeroBudget, err := execution.NewBudget(0)
	require.NoError(t, err)
	second, err := NewCachedGenerator(provider, CachedProviderOptions{
		Cache: cache, Limiter: zeroBudget, Workers: 1,
		AdapterVersion: "adapter-v1",
		ContextPolicy:  ContextPolicyNameForTest,
	})
	require.NoError(t, err)
	result, err := second.Generate(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, "answer", result.Text)
	require.Equal(t, int64(1), provider.calls.Load())
	require.Equal(t, execution.CacheHit, second.LastOutcome().State)
	require.Zero(t, zeroBudget.Spent())

	uncached := request
	uncached.Text = "new question"
	_, err = second.Generate(t.Context(), uncached)
	require.ErrorIs(t, err, execution.ErrBudgetExceeded)
	require.Equal(t, execution.CachePending, second.LastOutcome().State)
	require.Equal(t, int64(1), provider.calls.Load())
}

const ContextPolicyNameForTest = "whole-evidence-chunks-v1"
