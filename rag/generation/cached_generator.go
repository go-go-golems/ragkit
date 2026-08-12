package generation

import (
	"context"
	"sync"

	"github.com/go-go-golems/flowkit/execution"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// CachedGenerator decorates a Generator with item-level durable caching,
// limiting, and cumulative reporting.
type CachedGenerator struct {
	generator rag.Generator
	options   CachedProviderOptions
	mutex     sync.Mutex
	report    CachedProviderReport
	last      execution.CacheOutcome
}

var _ rag.Generator = (*CachedGenerator)(nil)

func NewCachedGenerator(
	generator rag.Generator,
	options CachedProviderOptions,
) (*CachedGenerator, error) {
	if generator == nil {
		return nil, errors.New("generator is required")
	}
	if err := validateCachedProviderOptions(options); err != nil {
		return nil, err
	}
	options.Workers = 1
	return &CachedGenerator{generator: generator, options: options}, nil
}

func (g *CachedGenerator) Generate(
	ctx context.Context,
	request rag.GenerationRequest,
) (rag.GenerationResult, error) {
	if g == nil {
		return rag.GenerationResult{}, errors.New("cached generator is required")
	}
	options := g.options
	outerCallback := options.OnGenerationResult
	options.OnGenerationResult = func(
		index int,
		result rag.GenerationResult,
		outcome execution.CacheOutcome,
	) error {
		g.mutex.Lock()
		g.last = outcome
		g.mutex.Unlock()
		if outerCallback != nil {
			return outerCallback(index, result, outcome)
		}
		return nil
	}
	results, report, err := GenerateCached(
		ctx, []rag.GenerationRequest{request}, options, g.generator,
	)
	g.mutex.Lock()
	g.report.Cache.Hits += report.Cache.Hits
	g.report.Cache.Misses += report.Cache.Misses
	g.report.Cache.Writes += report.Cache.Writes
	g.report.Cache.WorkCalls += report.Cache.WorkCalls
	g.report.Cache.Outcomes = append(
		g.report.Cache.Outcomes, report.Cache.Outcomes...,
	)
	g.report.Usage.Add(report.Usage)
	if len(report.Cache.Outcomes) > 0 {
		g.last = report.Cache.Outcomes[len(report.Cache.Outcomes)-1]
	}
	g.mutex.Unlock()
	if err != nil {
		return rag.GenerationResult{Usage: report.Usage}, err
	}
	return results[0], nil
}

func (g *CachedGenerator) Snapshot() CachedProviderReport {
	if g == nil {
		return CachedProviderReport{}
	}
	g.mutex.Lock()
	defer g.mutex.Unlock()
	report := g.report
	report.Usage = report.Usage.Clone()
	report.Cache.Outcomes = append(
		[]execution.CacheOutcome(nil), report.Cache.Outcomes...,
	)
	return report
}

func (g *CachedGenerator) LastOutcome() execution.CacheOutcome {
	if g == nil {
		return execution.CacheOutcome{}
	}
	g.mutex.Lock()
	defer g.mutex.Unlock()
	return g.last
}
