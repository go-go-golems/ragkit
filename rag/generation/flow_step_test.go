package generation

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/pkg/errors"
	"github.com/stretchr/testify/require"

	"github.com/go-go-golems/flowkit/execution"
	"github.com/go-go-golems/flowkit/flow"
	"github.com/go-go-golems/ragkit/rag"
)

type scriptedGenerator struct {
	calls atomic.Int64
	text  string
	fail  error
}

type usageErrorGenerator struct{}

func (usageErrorGenerator) Generate(context.Context, rag.GenerationRequest) (rag.GenerationResult, error) {
	tokens := int64(4)
	return rag.GenerationResult{Usage: rag.Usage{OutputTokens: &tokens}}, errors.New("provider failed")
}

func (generator *scriptedGenerator) Generate(_ context.Context, request rag.GenerationRequest) (rag.GenerationResult, error) {
	generator.calls.Add(1)
	if generator.fail != nil {
		return rag.GenerationResult{}, generator.fail
	}
	tokens := int64(7)
	reasoningTokens := int64(3)
	return rag.GenerationResult{
		Text:  generator.text,
		Usage: rag.Usage{OutputTokens: &tokens, ReasoningTokens: &reasoningTokens},
	}, nil
}

func flowStepRequest() rag.GenerationRequest {
	return rag.GenerationRequest{
		Kind:   "summary",
		Model:  "test-model",
		Prompt: "Summarize this.",
		Text:   "The corpus text.",
	}
}

// DR-3: the flow step's cache key must be byte-identical to the legacy
// generation cache key — same keyspace, same digest, same file path.
func TestFlowStepCacheKeyMatchesGenerationCacheKey(t *testing.T) {
	step, err := FlowStep(&scriptedGenerator{}, "answers", flow.Policy{}, "adapter-v3", "full-evidence")
	require.NoError(t, err)
	request := flowStepRequest()

	legacy, err := NewGenerationCacheKey(request, "adapter-v3", "full-evidence")
	require.NoError(t, err)
	viaFlow, err := step.Key(request)
	require.NoError(t, err)
	require.Equal(t, legacy, viaFlow)
	require.Equal(t, "generation", viaFlow.Step)
	require.Equal(t, "v1", viaFlow.Version)
}

// The stronger property behind DR-3: entries written by GenerateCached are
// cache hits for the flow step, and entries written by the flow step are
// cache hits for GenerateCached — one shared provider-steps cache across
// eras, in both directions.
func TestFlowStepReplaysGenerateCachedEntriesAndViceVersa(t *testing.T) {
	cache, err := execution.NewFileCache(execution.FileCacheOptions{Directory: t.TempDir()})
	require.NoError(t, err)
	request := flowStepRequest()

	// Era one: the legacy path stores an entry.
	legacyGenerator := &scriptedGenerator{text: "legacy answer"}
	legacyResults, _, err := GenerateCached(
		context.Background(),
		[]rag.GenerationRequest{request},
		CachedProviderOptions{Cache: cache, Workers: 1, AdapterVersion: "adapter-v3", ContextPolicy: "full-evidence"},
		legacyGenerator,
	)
	require.NoError(t, err)
	require.Equal(t, int64(1), legacyGenerator.calls.Load())

	// Era two: the flow step replays it without a provider call.
	flowGenerator := &scriptedGenerator{fail: errors.New("must not be called")}
	step, err := FlowStep(flowGenerator, "answers", flow.Policy{}, "adapter-v3", "full-evidence")
	require.NoError(t, err)
	results, report, err := flow.Run(
		context.Background(), step, []rag.GenerationRequest{request}, flow.Options{Store: cache},
	)
	require.NoError(t, err)
	require.Equal(t, int64(0), flowGenerator.calls.Load())
	require.Equal(t, 1, report.Step("answers").Hits)
	require.Equal(t, "legacy answer", results[0].Value.Result.Text)
	require.Equal(t, legacyResults[0].Text, Unwrap(results)[0].Text)

	// Reverse direction: a fresh request stored by the flow step is a hit
	// for GenerateCached.
	freshRequest := flowStepRequest()
	freshRequest.Text = "Different corpus text."
	freshGenerator := &scriptedGenerator{text: "flow answer"}
	freshStep, err := FlowStep(freshGenerator, "answers", flow.Policy{}, "adapter-v3", "full-evidence")
	require.NoError(t, err)
	_, _, err = flow.Run(
		context.Background(), freshStep, []rag.GenerationRequest{freshRequest}, flow.Options{Store: cache},
	)
	require.NoError(t, err)
	require.Equal(t, int64(1), freshGenerator.calls.Load())

	legacyReplay := &scriptedGenerator{fail: errors.New("must not be called")}
	replayed, replayReport, err := GenerateCached(
		context.Background(),
		[]rag.GenerationRequest{freshRequest},
		CachedProviderOptions{Cache: cache, Workers: 1, AdapterVersion: "adapter-v3", ContextPolicy: "full-evidence"},
		legacyReplay,
	)
	require.NoError(t, err)
	require.Equal(t, int64(0), legacyReplay.calls.Load())
	require.Equal(t, 1, replayReport.Cache.Hits)
	require.Equal(t, "flow answer", replayed[0].Text)
}

func TestFlowStepMetersFreshUsageOnly(t *testing.T) {
	cache := flow.NewMemoryStore()
	step, err := FlowStep(&scriptedGenerator{text: "answer"}, "metered", flow.Policy{}, "adapter-v3", "full-evidence")
	require.NoError(t, err)
	request := flowStepRequest()

	_, report, err := flow.Run(context.Background(), step, []rag.GenerationRequest{request}, flow.Options{Store: cache})
	require.NoError(t, err)
	require.Equal(t, flow.Meters{"output_tokens": 7, "reasoning_tokens": 3}, report.Step("metered").Meters)

	_, report, err = flow.Run(context.Background(), step, []rag.GenerationRequest{request}, flow.Options{Store: cache})
	require.NoError(t, err)
	require.Nil(t, report.Step("metered").Meters, "cached usage is not this run's spend")
}

func TestUsageMetersRoundTripsReasoningTokens(t *testing.T) {
	reasoningTokens := int64(13)
	usage := UsageFromMeters(UsageMeters(rag.Usage{ReasoningTokens: &reasoningTokens}))
	require.NotNil(t, usage.ReasoningTokens)
	require.Equal(t, reasoningTokens, *usage.ReasoningTokens)
}

func TestFlowStepMetersUsageReturnedWithFailure(t *testing.T) {
	step, err := FlowStep(usageErrorGenerator{}, "failed-generation", flow.Policy{}, "adapter-v1", "context-v1")
	require.NoError(t, err)
	_, report, err := flow.Run(t.Context(), step, []rag.GenerationRequest{flowStepRequest()}, flow.Options{})
	require.Error(t, err)
	require.Equal(t, flow.Meters{"output_tokens": 4}, report.Step("failed-generation").Meters)
}

func TestFlowStepValidatesOptionsLikeTheLegacyPath(t *testing.T) {
	_, err := FlowStep(nil, "x", flow.Policy{}, "v", "policy")
	require.Error(t, err)
	_, err = FlowStep(&scriptedGenerator{}, "", flow.Policy{}, "v", "policy")
	require.Error(t, err)
	_, err = FlowStep(&scriptedGenerator{}, "x", flow.Policy{}, "", "policy")
	require.ErrorContains(t, err, "adapter version")
	_, err = FlowStep(&scriptedGenerator{}, "x", flow.Policy{}, "v", "")
	require.ErrorContains(t, err, "context policy")
}

func TestUnwrapZeroesUsage(t *testing.T) {
	tokens := int64(11)
	results := []flow.Result[GenerationCacheEnvelope]{{
		Value: GenerationCacheEnvelope{Result: rag.GenerationResult{
			Text:  "hello",
			Usage: rag.Usage{OutputTokens: &tokens},
		}},
	}}
	unwrapped := Unwrap(results)
	require.Equal(t, "hello", unwrapped[0].Text)
	require.Equal(t, rag.Usage{}, unwrapped[0].Usage)
}
