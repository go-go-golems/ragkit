package generation

import (
	"context"
	"errors"
	"testing"

	"github.com/go-go-golems/ragkit/flow"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func TestFlowGeneratorReturnsAttemptUsageWithError(t *testing.T) {
	tokens := int64(11)
	provider := generatorFunc(func(context.Context, rag.GenerationRequest) (rag.GenerationResult, error) {
		return rag.GenerationResult{Usage: rag.Usage{OutputTokens: &tokens}}, errors.New("billed failure")
	})
	adapter, err := NewFlowGenerator(
		provider,
		"flow-generator-error-usage",
		flow.Policy{Workers: 1},
		"adapter-v1",
		ContextPolicyNameForTest,
		flow.Options{},
	)
	require.NoError(t, err)
	result, err := adapter.Generate(t.Context(), rag.GenerationRequest{Kind: "answer", Model: "model"})
	require.ErrorContains(t, err, "billed failure")
	require.NotNil(t, result.Usage.OutputTokens)
	require.EqualValues(t, 11, *result.Usage.OutputTokens)
	require.NotNil(t, adapter.Snapshot().Usage.OutputTokens)
	require.EqualValues(t, 11, *adapter.Snapshot().Usage.OutputTokens)
}
