package embedding

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/go-go-golems/flowkit/execution"
	"github.com/go-go-golems/ragkit/rag"
	vectorutil "github.com/go-go-golems/ragkit/vector"
	"github.com/pkg/errors"
)

// CachedEmbedderOptions controls recoverable, budgeted embedding calls.
type CachedEmbedderOptions struct {
	Cache   execution.Cache
	Limiter execution.Limiter
	Workers int
	Step    string
	Version string
}

// CachedEmbedderSnapshot reports cumulative item-level cache behavior.
type CachedEmbedderSnapshot struct {
	Hits      int64                  `json:"hits"`
	Misses    int64                  `json:"misses"`
	Writes    int64                  `json:"writes"`
	WorkCalls int64                  `json:"work_calls"`
	Usage     rag.Usage              `json:"usage"`
	Last      execution.CacheOutcome `json:"last"`
}

// CachedEmbedder applies cache lookup before resource admission for every
// individual text in an embedding request. It makes both indexing and query
// embeddings recoverable without coupling the cache to a provider.
type CachedEmbedder struct {
	embedder  rag.Embedder
	options   CachedEmbedderOptions
	hits      atomic.Int64
	misses    atomic.Int64
	writes    atomic.Int64
	workCalls atomic.Int64
	usage     rag.UsageAccumulator
	mutex     sync.Mutex
	last      execution.CacheOutcome
}

var _ rag.Embedder = (*CachedEmbedder)(nil)

func NewCachedEmbedder(embedder rag.Embedder, options CachedEmbedderOptions) (*CachedEmbedder, error) {
	if embedder == nil {
		return nil, errors.New("cached embedder requires an embedder")
	}
	if options.Cache == nil {
		return nil, errors.New("cached embedder requires a cache")
	}
	if options.Step == "" {
		options.Step = "embedding"
	}
	if options.Version == "" {
		options.Version = "v1"
	}
	if options.Workers < 1 {
		options.Workers = 1
	}
	return &CachedEmbedder{embedder: embedder, options: options}, nil
}

func (e *CachedEmbedder) Embed(ctx context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	if len(request.Texts) == 0 {
		return rag.EmbeddingResult{}, errors.New("embedding texts are required")
	}
	type item struct {
		Model string `json:"model"`
		Text  string `json:"text"`
	}
	items := make([]item, len(request.Texts))
	for index, text := range request.Texts {
		items[index] = item{Model: request.Model, Text: text}
	}
	var usage rag.UsageAccumulator
	var dimensionsMutex sync.Mutex
	dimensions := 0
	vectors, report, err := execution.MapCached(ctx, items, execution.CachedMapOptions[item]{
		Map:   execution.MapOptions[item]{Workers: e.options.Workers, Limiter: e.options.Limiter},
		Cache: e.options.Cache,
		Key: func(current item) (execution.Key, error) {
			return execution.NewKey(e.options.Step, e.options.Version, current)
		},
	}, func(ctx context.Context, current item) ([]float32, error) {
		result, err := e.embedder.Embed(ctx, rag.EmbeddingRequest{Model: current.Model, Texts: []string{current.Text}})
		// Provider work may be billed even when the call returns an error.
		usage.Add(result.Usage)
		if err != nil {
			return nil, err
		}
		// Provider work has already been billed even when its output later
		// fails shape or finite-value validation.
		if len(result.Vectors) != 1 {
			return nil, errors.Errorf("embedder returned %d vectors for one text", len(result.Vectors))
		}
		vector := result.Vectors[0]
		if len(vector) == 0 {
			return nil, errors.New("embedder returned an empty vector")
		}
		if err := vectorutil.ValidateFinite(vector); err != nil {
			return nil, errors.Wrap(err, "validate embedder vector")
		}
		dimensionsMutex.Lock()
		if dimensions == 0 {
			dimensions = len(vector)
		}
		wantDimensions := dimensions
		dimensionsMutex.Unlock()
		if len(vector) != wantDimensions {
			return nil, errors.Errorf("embedder changed vector dimensions within one request: got %d, want %d", len(vector), wantDimensions)
		}
		return vector, nil
	})
	e.hits.Add(int64(report.Hits))
	e.misses.Add(int64(report.Misses))
	e.writes.Add(int64(report.Writes))
	e.workCalls.Add(int64(report.WorkCalls))
	e.usage.Add(usage.Snapshot())
	if len(report.Outcomes) > 0 {
		e.mutex.Lock()
		e.last = report.Outcomes[len(report.Outcomes)-1]
		e.mutex.Unlock()
	}
	if err != nil {
		// MapCached cannot promise an aligned vector batch after one item fails,
		// but provider work before that failure may still have been billed.
		return rag.EmbeddingResult{Usage: usage.Snapshot()}, err
	}
	for index, vector := range vectors {
		if len(vector) == 0 {
			return rag.EmbeddingResult{}, fmt.Errorf("cached embedding vector %d is empty", index)
		}
		if index > 0 && len(vector) != len(vectors[0]) {
			return rag.EmbeddingResult{}, fmt.Errorf("embedding vector %d has %d dimensions, want %d", index, len(vector), len(vectors[0]))
		}
		if err := vectorutil.ValidateFinite(vector); err != nil {
			return rag.EmbeddingResult{}, errors.Wrapf(err, "validate embedding vector %d", index)
		}
	}
	return rag.EmbeddingResult{Vectors: vectors, Usage: usage.Snapshot()}, nil
}

func (e *CachedEmbedder) Snapshot() CachedEmbedderSnapshot {
	if e == nil {
		return CachedEmbedderSnapshot{}
	}
	e.mutex.Lock()
	last := e.last
	e.mutex.Unlock()
	return CachedEmbedderSnapshot{
		Hits: e.hits.Load(), Misses: e.misses.Load(), Writes: e.writes.Load(),
		WorkCalls: e.workCalls.Load(),
		Usage:     e.usage.Snapshot(),
		Last:      last,
	}
}
