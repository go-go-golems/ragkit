package generation

import (
	"context"
	"encoding/json"

	"github.com/go-go-golems/ragkit/execution"
	"github.com/go-go-golems/ragkit/flow"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// FlowStep builds the flow step for cached generation calls. Its identity
// reproduces the generation cache key EXACTLY — same "generation"/"v1"
// keyspace, same GenerationCacheKeyInput bytes, same GenerationCacheEnvelope
// value — so a flow run replays every entry the GenerateCached era stored
// and vice versa (RAG-TTC-FLOW-001 DR-3: wiring may change, identity may
// not). The adapter lives here, not in pkg/flow: flow stays generic, the
// domain adapts itself to it.
//
// Results carry the envelope as stored; use Unwrap for the GenerateCached
// return convention (usage zeroed — cached usage is not this run's spend;
// the step's Meter reports fresh usage instead).
func FlowStep(
	generator rag.Generator,
	name string,
	policy flow.Policy,
	adapterVersion string,
	contextPolicy string,
) (flow.Step[rag.GenerationRequest, GenerationCacheEnvelope], error) {
	var zero flow.Step[rag.GenerationRequest, GenerationCacheEnvelope]
	if generator == nil {
		return zero, errors.New("generation flow step requires a generator")
	}
	if name == "" {
		return zero, errors.New("generation flow step requires a name")
	}
	// Reuse the legacy validators so refusals match the cached paths.
	if err := validateCachedProviderOptions(CachedProviderOptions{
		Cache:          discardCache{},
		Workers:        1,
		AdapterVersion: adapterVersion,
		ContextPolicy:  contextPolicy,
	}); err != nil {
		return zero, err
	}
	return flow.Step[rag.GenerationRequest, GenerationCacheEnvelope]{
		Name: name,
		Identity: flow.Identity[rag.GenerationRequest]{
			Kind:    generationCacheStep,
			Version: generationCacheVersion,
			Key: func(request rag.GenerationRequest) ([]byte, error) {
				input, err := NewGenerationCacheKeyInput(request, adapterVersion, contextPolicy)
				if err != nil {
					return nil, err
				}
				return json.Marshal(input)
			},
		},
		Policy: policy,
		Do: func(ctx context.Context, request rag.GenerationRequest) (GenerationCacheEnvelope, error) {
			result, err := generator.Generate(ctx, request)
			if err != nil {
				return GenerationCacheEnvelope{}, err
			}
			return GenerationCacheEnvelope{Result: result}, nil
		},
		Meter: func(envelope GenerationCacheEnvelope) flow.Meters {
			return UsageMeters(envelope.Result.Usage)
		},
	}, nil
}

// Unwrap converts envelope results to generation results under the
// GenerateCached convention: per-item usage is zeroed, because cached usage
// was spent by whichever run stored the entry, not this one.
func Unwrap(results []flow.Result[GenerationCacheEnvelope]) []rag.GenerationResult {
	unwrapped := make([]rag.GenerationResult, len(results))
	for index, result := range results {
		unwrapped[index] = result.Value.Result
		unwrapped[index].Usage = rag.Usage{}
	}
	return unwrapped
}

// UsageMeters translates provider-reported usage into flow meters. Missing
// fields stay absent rather than reporting zero.
func UsageMeters(usage rag.Usage) flow.Meters {
	meters := flow.Meters{}
	if usage.InputTokens != nil {
		meters["input_tokens"] = float64(*usage.InputTokens)
	}
	if usage.CachedInputTokens != nil {
		meters["cached_input_tokens"] = float64(*usage.CachedInputTokens)
	}
	if usage.OutputTokens != nil {
		meters["output_tokens"] = float64(*usage.OutputTokens)
	}
	if usage.EmbeddingTokens != nil {
		meters["embedding_tokens"] = float64(*usage.EmbeddingTokens)
	}
	if usage.CostUSD != nil {
		meters["cost_usd"] = *usage.CostUSD
	}
	return meters
}

// discardCache satisfies the legacy validator's non-nil cache requirement
// during option validation; it is never used to load or store.
type discardCache struct{}

func (discardCache) Load(context.Context, execution.Key, any) (bool, error) { return false, nil }
func (discardCache) Store(context.Context, execution.Key, any) error        { return nil }
