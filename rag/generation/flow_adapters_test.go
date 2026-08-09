package generation

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/go-go-golems/ragkit/execution"
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

func TestFlowBatcherCountsAStartedSequenceWhenRetryAdmissionRefuses(t *testing.T) {
	provider := generatorFunc(func(context.Context, rag.GenerationRequest) (rag.GenerationResult, error) {
		return rag.GenerationResult{}, errors.New("timeout")
	})
	batcher, err := NewFlowBatcher(
		provider,
		"flow-batcher-refused-retry",
		flow.Policy{
			Workers:   1,
			Admission: []flow.Resource{{Name: "provider-calls", Ceiling: 2, Budget: 1}},
			Retry:     flow.RetrySpec{Attempts: 2, Backoff: flow.Backoff{Base: time.Millisecond, Cap: time.Millisecond}},
		},
		"adapter-v1",
		ContextPolicyNameForTest,
		flow.Options{},
	)
	require.NoError(t, err)
	_, err = batcher.Generate(t.Context(), []rag.GenerationRequest{{Kind: "answer", Model: "model"}})
	require.ErrorIs(t, err, execution.ErrBudgetExceeded)
	hits, workCalls := batcher.Counters()
	require.Zero(t, hits)
	require.Equal(t, 1, workCalls)

	report := CachedReportFromFlow(nil, flow.StepReport{Items: 1, Misses: 1, WorkCalls: 1, Retries: 1})
	require.Equal(t, 1, report.Cache.WorkCalls)
}
