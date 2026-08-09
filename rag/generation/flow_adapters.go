package generation

import (
	"context"
	"path/filepath"
	"sync"

	"github.com/go-go-golems/ragkit/execution"
	"github.com/go-go-golems/ragkit/flow"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// ProviderStepsDirectory is the shared cache subdirectory for provider-call
// caching (moved here from the retired pkg/harness). Every harness uses the
// same name so cached generations, embeddings, and rerankings replay across
// harnesses — the chunk-compare → answer-quality promotion seam depends on
// it.
const ProviderStepsDirectory = "provider-steps"

// OpenProviderSteps opens the canonical provider-steps cache beneath a cache
// root, as the flow durability store.
func OpenProviderSteps(cacheDirectory string) (*execution.FileCache, error) {
	if cacheDirectory == "" {
		return nil, errors.New("cache directory is required")
	}
	return execution.NewFileCache(execution.FileCacheOptions{
		Directory: filepath.Join(cacheDirectory, ProviderStepsDirectory),
	})
}

// UsageFromMeters is the inverse of UsageMeters: absent meter names stay nil
// fields, so reports keep the missing-vs-zero distinction.
func UsageFromMeters(meters flow.Meters) rag.Usage {
	usage := rag.Usage{}
	setInt := func(target **int64, name string) {
		if value, ok := meters[name]; ok {
			converted := int64(value)
			*target = &converted
		}
	}
	setInt(&usage.InputTokens, "input_tokens")
	setInt(&usage.CachedInputTokens, "cached_input_tokens")
	setInt(&usage.OutputTokens, "output_tokens")
	setInt(&usage.ReasoningTokens, "reasoning_tokens")
	setInt(&usage.EmbeddingTokens, "embedding_tokens")
	if value, ok := meters["cost_usd"]; ok {
		usage.CostUSD = &value
	}
	return usage
}

// CachedReportFromFlow reproduces the CachedProviderReport shape the
// GenerateCached era serialized into run summaries, from one flow step's
// results and report. The legacy report counts a work sequence only when at
// least one provider attempt ran; flow counts every physical attempt. A retry
// can be scheduled but then canceled or refused before a second Do call, so
// flow records a started sequence exactly when its first provider attempt
// begins.
func CachedReportFromFlow(results []flow.Result[GenerationCacheEnvelope], report flow.StepReport) CachedProviderReport {
	converted := CachedProviderReport{Usage: UsageFromMeters(report.Meters)}
	converted.Cache = execution.CacheReport{
		Hits:      report.Hits,
		Misses:    report.Misses,
		Writes:    report.Stored,
		WorkCalls: workSequences(report),
		Outcomes:  make([]execution.CacheOutcome, len(results)),
	}
	for index, result := range results {
		converted.Cache.Outcomes[index] = result.Cache
	}
	return converted
}

func workSequences(report flow.StepReport) int {
	return report.StartedSequences
}

// FlowBatcher implements the representations.BatchGenerate contract on the
// flow engine: every request runs through admission, retry classification,
// and the shared generation cache identity. One batcher accumulates cache
// counters across calls, replacing the per-harness cachedGeneration glue.
// The options are shared internally, so a multi-call arm draws every call
// from one budget.
type FlowBatcher struct {
	step    flow.Step[rag.GenerationRequest, GenerationCacheEnvelope]
	options flow.Options

	mutex     sync.Mutex
	hits      int
	workCalls int
	usage     rag.UsageAccumulator
}

// NewFlowBatcher builds a batcher over one generation flow step.
func NewFlowBatcher(
	generator rag.Generator,
	name string,
	policy flow.Policy,
	adapterVersion string,
	contextPolicy string,
	options flow.Options,
) (*FlowBatcher, error) {
	step, err := FlowStep(generator, name, policy, adapterVersion, contextPolicy)
	if err != nil {
		return nil, err
	}
	return &FlowBatcher{step: step, options: options.Share()}, nil
}

// Generate runs one batch of requests, in the BatchGenerate shape.
func (batcher *FlowBatcher) Generate(ctx context.Context, requests []rag.GenerationRequest) ([]rag.GenerationResult, error) {
	results, _, err := batcher.generate(ctx, requests)
	return results, err
}

// generate runs the batch and returns the usage charged by this invocation.
// The internal helper lets FlowGenerator preserve billed usage on an error
// without returning the batcher's cumulative ledger as if it were per-call.
func (batcher *FlowBatcher) generate(ctx context.Context, requests []rag.GenerationRequest) ([]rag.GenerationResult, rag.Usage, error) {
	results, report, err := flow.Run(ctx, batcher.step, requests, batcher.options)
	stepReport := report.Step(batcher.step.Name)
	freshUsage := UsageFromMeters(stepReport.Meters)
	batcher.mutex.Lock()
	batcher.hits += stepReport.Hits
	batcher.workCalls += workSequences(stepReport)
	batcher.usage.Add(freshUsage)
	batcher.mutex.Unlock()
	if err != nil {
		return nil, freshUsage, err
	}
	return Unwrap(results), freshUsage, nil
}

// Counters returns the accumulated cache hits and work calls (legacy
// semantics: one work call per completed retry sequence).
func (batcher *FlowBatcher) Counters() (int, int) {
	batcher.mutex.Lock()
	defer batcher.mutex.Unlock()
	return batcher.hits, batcher.workCalls
}

// Usage returns the accumulated fresh-work usage.
func (batcher *FlowBatcher) Usage() rag.Usage {
	return batcher.usage.Snapshot()
}

// FlowGenerator adapts one generation flow step to the rag.Generator
// contract for per-call consumers (the query-strategy rewrites), replacing
// NewCachedGenerator + WithRetry. Calls share the batcher options' budgets.
type FlowGenerator struct {
	batcher *FlowBatcher
}

var _ rag.Generator = (*FlowGenerator)(nil)

// NewFlowGenerator builds the per-call adapter.
func NewFlowGenerator(
	generator rag.Generator,
	name string,
	policy flow.Policy,
	adapterVersion string,
	contextPolicy string,
	options flow.Options,
) (*FlowGenerator, error) {
	batcher, err := NewFlowBatcher(generator, name, policy, adapterVersion, contextPolicy, options)
	if err != nil {
		return nil, err
	}
	return &FlowGenerator{batcher: batcher}, nil
}

// Generate implements rag.Generator.
func (adapter *FlowGenerator) Generate(ctx context.Context, request rag.GenerationRequest) (rag.GenerationResult, error) {
	results, usage, err := adapter.batcher.generate(ctx, []rag.GenerationRequest{request})
	if err != nil {
		return rag.GenerationResult{Usage: usage}, err
	}
	return results[0], nil
}

// Snapshot returns the accumulated report in the legacy shape.
func (adapter *FlowGenerator) Snapshot() CachedProviderReport {
	hits, workCalls := adapter.batcher.Counters()
	return CachedProviderReport{
		Cache: execution.CacheReport{Hits: hits, WorkCalls: workCalls},
		Usage: adapter.batcher.Usage(),
	}
}
