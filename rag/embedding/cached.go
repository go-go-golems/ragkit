package embedding

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/execution"
	"github.com/go-go-golems/ragkit/flow"
	"github.com/go-go-golems/ragkit/rag"
	vectorutil "github.com/go-go-golems/ragkit/vector"
)

// Item is one stable, content-addressed input to a corpus embedding operation.
type Item struct {
	ID            string
	Text          string
	ContentDigest string
}

// CachedOptions controls bounded, recoverable corpus embedding on the flow
// engine. Step/Version name the cache family exactly as before, so every
// vector the pre-flow era stored replays.
type CachedOptions struct {
	Model     string
	Workers   int
	BatchSize int
	Step      string
	Version   string
	// Resource is the fail-closed admission plan the items draw from. Pass
	// the full plan, or a name-only reference to a plan already declared on
	// the shared Flow options. Empty name means unbudgeted embedding.
	Resource flow.Resource
	// Retry bounds retries around each provider batch — the coverage whose
	// absence killed an otherwise-complete 13,847-call build at embeddings
	// item 49 (2026-07-31).
	Retry flow.RetrySpec
	// Flow carries the durable store (and optionally shared budgets).
	Flow flow.Options
}

// CachedResult contains vectors in input order and accounting for provider
// work performed during this invocation. Usage excludes cache hits.
type CachedResult struct {
	Vectors []rag.Vector
	Usage   rag.Usage
	Cache   execution.CacheReport
}

// cachedKeyInput mirrors the pre-flow cache key input byte for byte.
type cachedKeyInput struct {
	Model         string `json:"model"`
	ItemID        string `json:"item_id"`
	ContentDigest string `json:"content_digest"`
}

// Cached embeds identified items in bounded batches while caching every
// vector independently. Completed batches remain recoverable after a later
// failure; transient provider errors retry per Retry.
func Cached(
	ctx context.Context,
	embedder rag.Embedder,
	items []Item,
	options CachedOptions,
) (CachedResult, error) {
	if embedder == nil {
		return CachedResult{}, fmt.Errorf("embedder is required")
	}
	if options.Flow.Store == nil {
		return CachedResult{}, fmt.Errorf("embedding cache store is required")
	}
	if options.Model == "" {
		return CachedResult{}, fmt.Errorf("embedding model is required")
	}
	if options.Workers < 1 {
		return CachedResult{}, fmt.Errorf("embedding workers must be positive")
	}
	if options.BatchSize < 1 {
		return CachedResult{}, fmt.Errorf("embedding batch size must be positive")
	}
	if options.Step == "" {
		return CachedResult{}, fmt.Errorf("embedding cache step is required")
	}
	if options.Version == "" {
		return CachedResult{}, fmt.Errorf("embedding cache version is required")
	}
	for index, item := range items {
		if item.ID == "" {
			return CachedResult{}, fmt.Errorf("embedding item %d: id is required", index)
		}
		if item.ContentDigest == "" {
			return CachedResult{}, fmt.Errorf("embedding item %d: content digest is required", index)
		}
		if actual := digest.Text(item.Text); actual != item.ContentDigest {
			return CachedResult{}, fmt.Errorf("embedding item %d: content digest mismatch: stored=%s actual=%s", index, item.ContentDigest, actual)
		}
	}

	var usage rag.UsageAccumulator
	var dimensions struct {
		sync.Mutex
		value int
	}
	base := flow.Step[Item, rag.Vector]{
		Name: options.Step,
		Identity: flow.Identity[Item]{
			Kind:    options.Step,
			Version: options.Version,
			Key: func(item Item) ([]byte, error) {
				return json.Marshal(cachedKeyInput{
					Model: options.Model, ItemID: item.ID, ContentDigest: item.ContentDigest,
				})
			},
		},
		Policy: flow.Policy{
			Workers: options.Workers,
			Retry:   options.Retry,
		},
	}
	if options.Resource.Name != "" {
		base.Policy.Admission = []flow.Resource{options.Resource}
	}
	step := flow.Bulk(base, func(ctx context.Context, batch []Item) ([]rag.Vector, error) {
		texts := make([]string, len(batch))
		for index, item := range batch {
			texts[index] = item.Text
		}
		result, err := embedder.Embed(ctx, rag.EmbeddingRequest{Model: options.Model, Texts: texts})
		usage.Add(result.Usage)
		if err != nil {
			return nil, err
		}
		if len(result.Vectors) != len(batch) {
			return nil, fmt.Errorf(
				"embedder returned %d vectors for %d items",
				len(result.Vectors),
				len(batch),
			)
		}
		batchDimensions, err := validateProviderVectors(result.Vectors)
		if err != nil {
			return nil, err
		}
		dimensions.Lock()
		if dimensions.value == 0 {
			dimensions.value = batchDimensions
		} else if dimensions.value != batchDimensions {
			err = fmt.Errorf(
				"embedding vector dimensions differ across batches: got %d, want %d",
				batchDimensions,
				dimensions.value,
			)
		}
		dimensions.Unlock()
		if err != nil {
			return nil, err
		}

		batchVectors := make([]rag.Vector, len(batch))
		for index, values := range result.Vectors {
			batchVectors[index] = rag.Vector{
				RepresentationID: batch[index].ID,
				Values:           values,
				Model:            options.Model,
			}
		}
		return batchVectors, nil
	}, options.BatchSize)

	results, flowReport, err := flow.Run(ctx, step, items, options.Flow)
	stepReport := flowReport.Step(options.Step)
	cachedResult := CachedResult{Usage: usage.Snapshot()}
	cachedResult.Cache = execution.CacheReport{
		Hits:      stepReport.Hits,
		Misses:    stepReport.Misses,
		Writes:    stepReport.Stored,
		WorkCalls: stepReport.WorkCalls - stepReport.Retries,
		Outcomes:  make([]execution.CacheOutcome, len(results)),
	}
	vectors := make([]rag.Vector, len(results))
	for index, result := range results {
		vectors[index] = result.Value
		cachedResult.Cache.Outcomes[index] = result.Cache
	}
	cachedResult.Vectors = vectors
	if err != nil {
		return cachedResult, err
	}
	if err := validateCachedVectors(items, options.Model, vectors); err != nil {
		return cachedResult, err
	}
	return cachedResult, nil
}

func validateProviderVectors(vectors [][]float32) (int, error) {
	dimensions := 0
	for index, values := range vectors {
		if len(values) == 0 {
			return 0, fmt.Errorf("embedding vector %d is empty", index)
		}
		if dimensions == 0 {
			dimensions = len(values)
		}
		if err := vectorutil.ValidateDimensions(dimensions, values); err != nil {
			return 0, fmt.Errorf("embedding vector %d: %w", index, err)
		}
		if err := vectorutil.ValidateFinite(values); err != nil {
			return 0, fmt.Errorf("embedding vector %d: %w", index, err)
		}
	}
	return dimensions, nil
}

func validateCachedVectors(items []Item, model string, vectors []rag.Vector) error {
	if len(vectors) != len(items) {
		return fmt.Errorf("embedding result has %d vectors for %d items", len(vectors), len(items))
	}
	dimensions := 0
	for index, current := range vectors {
		if current.RepresentationID != items[index].ID {
			return fmt.Errorf(
				"embedding vector %d has item id %q, want %q",
				index,
				current.RepresentationID,
				items[index].ID,
			)
		}
		if current.Model != model {
			return fmt.Errorf(
				"embedding vector %d has model %q, want %q",
				index,
				current.Model,
				model,
			)
		}
		if len(current.Values) == 0 {
			return fmt.Errorf("embedding vector %d is empty", index)
		}
		if dimensions == 0 {
			dimensions = len(current.Values)
		}
		if err := vectorutil.ValidateDimensions(dimensions, current.Values); err != nil {
			return fmt.Errorf("embedding vector %d: %w", index, err)
		}
		if err := vectorutil.ValidateFinite(current.Values); err != nil {
			return fmt.Errorf("embedding vector %d: %w", index, err)
		}
	}
	return nil
}
