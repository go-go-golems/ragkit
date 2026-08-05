package generation

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func TestObservedGeneratorSuccessAndDisclosure(t *testing.T) {
	t.Parallel()
	input, output := int64(4), int64(2)
	provider := generatorFunc(func(_ context.Context, _ rag.GenerationRequest) (rag.GenerationResult, error) {
		return rag.GenerationResult{
			Text: "answer", FinishReason: "stop",
			Usage: rag.Usage{InputTokens: &input, OutputTokens: &output},
		}, nil
	})
	var observed GenerationObservation
	generator, err := ObservedGenerator(provider, ObservationPolicy{
		IncludePrompt: true, IncludeText: true, IncludeResponse: true,
	}, func(_ context.Context, observation GenerationObservation) error {
		observed = observation
		return nil
	})
	require.NoError(t, err)
	request := rag.GenerationRequest{
		Kind: "answer", Model: "model", Prompt: "prompt", Text: "query",
		OutputSchema: "schema",
		Evidence:     []rag.Evidence{{Chunk: rag.Chunk{ID: "secret-evidence"}}},
	}
	result, err := generator.Generate(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, "answer", result.Text)
	require.Equal(t, "answer", observed.RequestKind)
	require.Equal(t, "model", observed.Model)
	require.Equal(t, "stop", observed.FinishReason)
	require.NotZero(t, observed.StartedAt)
	require.GreaterOrEqual(t, observed.Duration, time.Duration(0))
	require.NotNil(t, observed.Request)
	require.Equal(t, "prompt", observed.Request.Prompt)
	require.Equal(t, "query", observed.Request.Text)
	require.Equal(t, "schema", observed.Request.OutputSchema)
	require.Empty(t, observed.Request.Evidence)
	require.NotNil(t, observed.Response)
	require.Equal(t, "answer", observed.Response.Text)
}

func TestObservedGeneratorDefaultsToMetadataOnly(t *testing.T) {
	t.Parallel()
	var observed GenerationObservation
	generator, err := ObservedGenerator(
		generatorFunc(func(_ context.Context, _ rag.GenerationRequest) (rag.GenerationResult, error) {
			return rag.GenerationResult{Text: "private"}, nil
		}),
		ObservationPolicy{},
		func(_ context.Context, observation GenerationObservation) error {
			observed = observation
			return nil
		},
	)
	require.NoError(t, err)
	_, err = generator.Generate(t.Context(), rag.GenerationRequest{
		Kind: "kind", Model: "model", Prompt: "private", Text: "private",
	})
	require.NoError(t, err)
	require.Nil(t, observed.Request)
	require.Nil(t, observed.Response)
}

func TestObservedGeneratorProviderAndObserverErrors(t *testing.T) {
	t.Parallel()
	providerErr := errors.New("provider failed")
	observerErr := errors.New("observer failed")
	t.Run("provider error is observed", func(t *testing.T) {
		var observed GenerationObservation
		generator, err := ObservedGenerator(
			generatorFunc(func(context.Context, rag.GenerationRequest) (rag.GenerationResult, error) {
				return rag.GenerationResult{}, providerErr
			}),
			ObservationPolicy{IncludeResponse: true},
			func(_ context.Context, observation GenerationObservation) error {
				observed = observation
				return nil
			},
		)
		require.NoError(t, err)
		_, err = generator.Generate(t.Context(), rag.GenerationRequest{Kind: "kind", Model: "model"})
		require.ErrorIs(t, err, providerErr)
		require.Equal(t, providerErr.Error(), observed.ProviderError)
		require.Nil(t, observed.Response)
	})
	t.Run("observer error fails call", func(t *testing.T) {
		generator, err := ObservedGenerator(
			generatorFunc(func(context.Context, rag.GenerationRequest) (rag.GenerationResult, error) {
				return rag.GenerationResult{Text: "answer"}, nil
			}),
			ObservationPolicy{},
			func(context.Context, GenerationObservation) error { return observerErr },
		)
		require.NoError(t, err)
		_, err = generator.Generate(t.Context(), rag.GenerationRequest{})
		require.ErrorIs(t, err, observerErr)
	})
}

func TestObservedGeneratorObservesCancellationWithDetachedContext(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	var observerContextError error
	generator, err := ObservedGenerator(
		generatorFunc(func(ctx context.Context, _ rag.GenerationRequest) (rag.GenerationResult, error) {
			return rag.GenerationResult{}, ctx.Err()
		}),
		ObservationPolicy{},
		func(ctx context.Context, observation GenerationObservation) error {
			observerContextError = ctx.Err()
			require.Equal(t, context.Canceled.Error(), observation.ProviderError)
			return nil
		},
	)
	require.NoError(t, err)
	_, err = generator.Generate(ctx, rag.GenerationRequest{})
	require.ErrorIs(t, err, context.Canceled)
	require.NoError(t, observerContextError)
}

type generatorFunc func(context.Context, rag.GenerationRequest) (rag.GenerationResult, error)

func (function generatorFunc) Generate(
	ctx context.Context,
	request rag.GenerationRequest,
) (rag.GenerationResult, error) {
	return function(ctx, request)
}
