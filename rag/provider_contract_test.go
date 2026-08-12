package rag_test

import (
	"context"
	"errors"
	"testing"

	"github.com/go-go-golems/flowkit/execution"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/go-go-golems/ragkit/rag/embedding"
	"github.com/go-go-golems/ragkit/rag/generation"
	"github.com/go-go-golems/ragkit/rag/reranking"
	"github.com/stretchr/testify/require"
)

type generatorFunc func(context.Context, rag.GenerationRequest) (rag.GenerationResult, error)

func (f generatorFunc) Generate(ctx context.Context, request rag.GenerationRequest) (rag.GenerationResult, error) {
	return f(ctx, request)
}

type embedderFunc func(context.Context, rag.EmbeddingRequest) (rag.EmbeddingResult, error)

func (f embedderFunc) Embed(ctx context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	return f(ctx, request)
}

type rerankerFunc func(context.Context, rag.RerankRequest) (rag.RerankResult, error)

func (f rerankerFunc) Rerank(ctx context.Context, request rag.RerankRequest) (rag.RerankResult, error) {
	return f(ctx, request)
}

// requireUsagePreserved is the reusable provider-decorator contract. A
// decorator may omit an incomplete payload on error, but it must expose the
// exact charged usage once through its returned provider result.
func requireUsagePreserved(t *testing.T, want rag.Usage, invoke func() (rag.Usage, error)) {
	t.Helper()
	got, err := invoke()
	require.Error(t, err)
	require.Equal(t, want, got)
}

func TestProviderDecoratorsPreserveChargedUsageOnError(t *testing.T) {
	providerErr := errors.New("provider failed after billing")

	t.Run("observed generator", func(t *testing.T) {
		want := rag.Usage{OutputTokens: int64Pointer(7)}
		decorator, err := generation.ObservedGenerator(
			generatorFunc(func(context.Context, rag.GenerationRequest) (rag.GenerationResult, error) {
				return rag.GenerationResult{Usage: want}, providerErr
			}),
			generation.ObservationPolicy{},
			func(context.Context, generation.GenerationObservation) error { return nil },
		)
		require.NoError(t, err)
		requireUsagePreserved(t, want, func() (rag.Usage, error) {
			result, err := decorator.Generate(t.Context(), rag.GenerationRequest{})
			return result.Usage, err
		})
	})

	t.Run("cached generator", func(t *testing.T) {
		want := rag.Usage{OutputTokens: int64Pointer(7)}
		cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
		require.NoError(t, err)
		decorator, err := generation.NewCachedGenerator(
			generatorFunc(func(context.Context, rag.GenerationRequest) (rag.GenerationResult, error) {
				return rag.GenerationResult{Usage: want}, providerErr
			}),
			generation.CachedProviderOptions{Cache: cache, Workers: 1, AdapterVersion: "contract-test", ContextPolicy: "contract-test"},
		)
		require.NoError(t, err)
		requireUsagePreserved(t, want, func() (rag.Usage, error) {
			result, err := decorator.Generate(t.Context(), rag.GenerationRequest{})
			return result.Usage, err
		})
	})

	t.Run("cached embedder", func(t *testing.T) {
		want := rag.Usage{EmbeddingTokens: int64Pointer(7)}
		cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
		require.NoError(t, err)
		decorator, err := embedding.NewCachedEmbedder(
			embedderFunc(func(context.Context, rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
				return rag.EmbeddingResult{Usage: want}, providerErr
			}),
			embedding.CachedEmbedderOptions{Cache: cache},
		)
		require.NoError(t, err)
		requireUsagePreserved(t, want, func() (rag.Usage, error) {
			result, err := decorator.Embed(t.Context(), rag.EmbeddingRequest{Texts: []string{"text"}})
			return result.Usage, err
		})
	})

	t.Run("cached reranker", func(t *testing.T) {
		want := rag.Usage{InputTokens: int64Pointer(7)}
		cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
		require.NoError(t, err)
		decorator, err := reranking.NewCachedReranker(
			rerankerFunc(func(context.Context, rag.RerankRequest) (rag.RerankResult, error) {
				return rag.RerankResult{Usage: want}, providerErr
			}),
			reranking.CachedOptions{Cache: cache, Workers: 1, AdapterVersion: "contract-test"},
		)
		require.NoError(t, err)
		requireUsagePreserved(t, want, func() (rag.Usage, error) {
			result, err := decorator.Rerank(t.Context(), rag.RerankRequest{})
			return result.Usage, err
		})
	})
}

func int64Pointer(value int64) *int64 { return &value }
