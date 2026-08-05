package embedding

import (
	"context"
	"fmt"

	"github.com/go-go-golems/ragkit/rag"
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
	for start := 0; start < len(representations); start += batchSize {
		end := min(start+batchSize, len(representations))
		texts := make([]string, end-start)
		for index, representation := range representations[start:end] {
			texts[index] = representation.Text
		}
		result, err := embedder.Embed(ctx, rag.EmbeddingRequest{Model: model, Texts: texts})
		if err != nil {
			return nil, rag.Usage{}, fmt.Errorf("embed batch %d: %w", start/batchSize, err)
		}
		if len(result.Vectors) != len(texts) {
			return nil, rag.Usage{}, fmt.Errorf("embed batch %d returned %d vectors for %d texts", start/batchSize, len(result.Vectors), len(texts))
		}
		dimensions := 0
		for index, values := range result.Vectors {
			if dimensions == 0 {
				dimensions = len(values)
			}
			if len(values) == 0 || len(values) != dimensions {
				return nil, rag.Usage{}, fmt.Errorf("embed batch %d has invalid vector dimensions", start/batchSize)
			}
			vectors = append(vectors, rag.Vector{
				RepresentationID: representations[start+index].ID,
				Values:           values,
				Model:            model,
			})
		}
		usage.Add(result.Usage)
	}
	return vectors, usage, nil
}
