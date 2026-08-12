package generation

import (
	"context"
	"strings"

	"github.com/go-go-golems/flowkit/execution"
	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

const (
	generationCacheStep    = "generation"
	generationCacheVersion = "v1"
)

// GenerationCacheKeyInput covers every semantic input that can affect a
// generation result.
type GenerationCacheKeyInput struct {
	Model          string                 `json:"model"`
	Kind           string                 `json:"kind"`
	QueryDigest    string                 `json:"query_digest"`
	Evidence       []rag.EvidenceIdentity `json:"evidence"`
	PromptDigest   string                 `json:"prompt_digest"`
	SchemaDigest   string                 `json:"schema_digest"`
	AdapterVersion string                 `json:"adapter_version"`
	ContextPolicy  string                 `json:"context_policy"`
}

type GenerationCacheEnvelope struct {
	Result rag.GenerationResult `json:"result"`
}

type CachedProviderOptions struct {
	Cache              execution.Cache
	Limiter            execution.Limiter
	Workers            int
	AdapterVersion     string
	ContextPolicy      string
	OnGenerationResult func(
		index int,
		result rag.GenerationResult,
		outcome execution.CacheOutcome,
	) error
}

type CachedProviderReport struct {
	Cache execution.CacheReport `json:"cache"`
	Usage rag.Usage             `json:"usage"`
}

// NewGenerationCacheKeyInput assembles the semantic identity of one
// generation call. Both the legacy cached paths and the flow step derive
// their cache keys from this one struct, so their keys agree byte for byte.
func NewGenerationCacheKeyInput(request rag.GenerationRequest, adapterVersion, contextPolicy string) (GenerationCacheKeyInput, error) {
	if strings.TrimSpace(adapterVersion) == "" {
		return GenerationCacheKeyInput{}, errors.New("generation adapter version is required")
	}
	queryDigest, err := digest.JSON(request.Text)
	if err != nil {
		return GenerationCacheKeyInput{}, errors.Wrap(err, "digest generation query")
	}
	promptDigest, err := digest.JSON(request.Prompt)
	if err != nil {
		return GenerationCacheKeyInput{}, errors.Wrap(err, "digest generation prompt")
	}
	schemaDigest, err := digest.JSON(request.OutputSchema)
	if err != nil {
		return GenerationCacheKeyInput{}, errors.Wrap(err, "digest generation schema")
	}
	evidence, err := rag.EvidenceIdentities(request.Evidence)
	if err != nil {
		return GenerationCacheKeyInput{}, err
	}
	return GenerationCacheKeyInput{
		Model: request.Model, Kind: request.Kind, QueryDigest: queryDigest,
		Evidence: evidence, PromptDigest: promptDigest, SchemaDigest: schemaDigest,
		AdapterVersion: adapterVersion, ContextPolicy: contextPolicy,
	}, nil
}

func NewGenerationCacheKey(request rag.GenerationRequest, adapterVersion, contextPolicy string) (execution.Key, error) {
	input, err := NewGenerationCacheKeyInput(request, adapterVersion, contextPolicy)
	if err != nil {
		return execution.Key{}, err
	}
	return execution.NewKey(generationCacheStep, generationCacheVersion, input)
}

func GenerateCached(
	ctx context.Context,
	requests []rag.GenerationRequest,
	options CachedProviderOptions,
	generator rag.Generator,
) ([]rag.GenerationResult, CachedProviderReport, error) {
	report := CachedProviderReport{}
	if generator == nil {
		return nil, report, errors.New("generator is required")
	}
	if err := validateCachedProviderOptions(options); err != nil {
		return nil, report, err
	}
	var usage rag.UsageAccumulator
	var callback []func(int, GenerationCacheEnvelope, execution.CacheOutcome) error
	if options.OnGenerationResult != nil {
		callback = append(callback, func(index int, envelope GenerationCacheEnvelope, outcome execution.CacheOutcome) error {
			return options.OnGenerationResult(index, envelope.Result, outcome)
		})
	}
	envelopes, cacheReport, err := execution.MapCached(ctx, requests, execution.CachedMapOptions[rag.GenerationRequest]{
		Map:   execution.MapOptions[rag.GenerationRequest]{Workers: options.Workers, Limiter: options.Limiter},
		Cache: options.Cache,
		Key: func(request rag.GenerationRequest) (execution.Key, error) {
			return NewGenerationCacheKey(request, options.AdapterVersion, options.ContextPolicy)
		},
	}, func(ctx context.Context, request rag.GenerationRequest) (GenerationCacheEnvelope, error) {
		result, err := generator.Generate(ctx, request)
		usage.Add(result.Usage)
		if err != nil {
			return GenerationCacheEnvelope{}, err
		}
		return GenerationCacheEnvelope{Result: result}, nil
	}, callback...)
	report.Cache = cacheReport
	report.Usage = usage.Snapshot()
	if err != nil {
		return nil, report, err
	}
	results := make([]rag.GenerationResult, len(envelopes))
	for i, envelope := range envelopes {
		results[i] = envelope.Result
		results[i].Usage = rag.Usage{}
	}
	return results, report, nil
}

func validateCachedProviderOptions(options CachedProviderOptions) error {
	if options.Cache == nil {
		return errors.New("provider cache is required")
	}
	if options.Workers < 1 {
		return errors.New("provider cache workers must be positive")
	}
	if strings.TrimSpace(options.AdapterVersion) == "" {
		return errors.New("provider adapter version is required")
	}
	if strings.TrimSpace(options.ContextPolicy) == "" {
		return errors.New("generation context policy is required")
	}
	return nil
}
