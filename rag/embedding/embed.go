package embedding

import (
	"context"
	"fmt"

	"github.com/go-go-golems/ragkit/rag"
	vectorutil "github.com/go-go-golems/ragkit/vector"
)

// Representations embeds representations in explicit provider batches.
func Representations(
	ctx context.Context,
	embedder rag.Embedder,
	model string,
	representations []rag.Representation,
	batchSize int,
) ([]rag.Vector, rag.Usage, error) {
	if embedder == nil {
		return nil, rag.Usage{}, fmt.Errorf("embedder is required")
	}
	if batchSize < 1 {
		return nil, rag.Usage{}, fmt.Errorf("embedding batch size must be positive")
	}
	vectors := make([]rag.Vector, 0, len(representations))
	var usage rag.Usage
	dimensions := 0
	for start := 0; start < len(representations); start += batchSize {
		end := min(start+batchSize, len(representations))
		texts := make([]string, end-start)
		for index, representation := range representations[start:end] {
			texts[index] = representation.Text
		}
		result, err := embedder.Embed(ctx, rag.EmbeddingRequest{Model: model, Texts: texts})
		usage.Add(result.Usage)
		if err != nil {
			return nil, usage, fmt.Errorf("embed batch %d: %w", start/batchSize, err)
		}
		if len(result.Vectors) != len(texts) {
			return nil, usage, fmt.Errorf("embed batch %d returned %d vectors for %d texts", start/batchSize, len(result.Vectors), len(texts))
		}
		for index, values := range result.Vectors {
			if err := vectorutil.ValidateFinite(values); err != nil {
				return nil, usage, fmt.Errorf("embed batch %d vector %d: %w", start/batchSize, index, err)
			}
			if dimensions == 0 {
				dimensions = len(values)
			}
			if len(values) == 0 || len(values) != dimensions {
				return nil, usage, fmt.Errorf("embed batch %d has invalid vector dimensions: got %d, want %d", start/batchSize, len(values), dimensions)
			}
			vectors = append(vectors, rag.Vector{
				RepresentationID: representations[start+index].ID,
				Values:           values,
				Model:            model,
			})
		}
	}
	return vectors, usage, nil
}
