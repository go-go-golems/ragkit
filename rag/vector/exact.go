package vector

import (
	"context"
	"fmt"
	"sort"

	"github.com/go-go-golems/ragkit/rag"
	vectorutil "github.com/go-go-golems/ragkit/vector"
)

// Exact performs exhaustive cosine search.
type Exact struct {
	model           string
	channel         string
	queryEmbedder   rag.Embedder
	representations map[string]rag.Representation
	chunks          map[string]rag.Chunk
	vectors         []rag.Vector
}

var _ rag.Index = (*Exact)(nil)

// NewExact validates and builds an exact index.
func NewExact(
	model string,
	channel string,
	queryEmbedder rag.Embedder,
	representations []rag.Representation,
	chunks []rag.Chunk,
	vectors []rag.Vector,
) (*Exact, error) {
	if queryEmbedder == nil {
		return nil, fmt.Errorf("query embedder is required")
	}
	if channel == "" {
		channel = "vector"
	}
	representationByID := make(map[string]rag.Representation, len(representations))
	for _, representation := range representations {
		representationByID[representation.ID] = representation
	}
	chunkByID := make(map[string]rag.Chunk, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ID] = chunk
	}
	dimensions := 0
	for _, vector := range vectors {
		if vector.Model != model {
			return nil, fmt.Errorf("vector %q model %q does not match index model %q", vector.RepresentationID, vector.Model, model)
		}
		representation, ok := representationByID[vector.RepresentationID]
		if !ok {
			return nil, fmt.Errorf("vector references unknown representation %q", vector.RepresentationID)
		}
		if _, ok := chunkByID[representation.ChunkID]; !ok {
			return nil, fmt.Errorf("representation references unknown chunk %q", representation.ChunkID)
		}
		if dimensions == 0 {
			dimensions = len(vector.Values)
		}
		if len(vector.Values) == 0 || len(vector.Values) != dimensions {
			return nil, fmt.Errorf("inconsistent vector dimensions")
		}
		if err := vectorutil.ValidateFinite(vector.Values); err != nil {
			return nil, fmt.Errorf("vector %q: %w", vector.RepresentationID, err)
		}
	}
	ownedVectors := make([]rag.Vector, len(vectors))
	for index, vector := range vectors {
		ownedVectors[index] = vector
		ownedVectors[index].Values = append([]float32(nil), vector.Values...)
	}
	return &Exact{
		model:           model,
		channel:         channel,
		queryEmbedder:   queryEmbedder,
		representations: representationByID,
		chunks:          chunkByID,
		vectors:         ownedVectors,
	}, nil
}

func (index *Exact) Search(ctx context.Context, query rag.Query, limit int) ([]rag.Hit, error) {
	if index == nil {
		return nil, fmt.Errorf("vector index is nil")
	}
	if limit < 1 {
		return nil, fmt.Errorf("search limit must be positive")
	}
	embedded, err := index.queryEmbedder.Embed(ctx, rag.EmbeddingRequest{Model: index.model, Texts: []string{query.Text}})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(embedded.Vectors) != 1 {
		return nil, fmt.Errorf("query embedder returned %d vectors", len(embedded.Vectors))
	}
	queryVector := embedded.Vectors[0]
	if err := vectorutil.ValidateFinite(queryVector); err != nil {
		return nil, fmt.Errorf("query vector: %w", err)
	}
	hits := make([]rag.Hit, 0, len(index.vectors))
	for _, vector := range index.vectors {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		score, err := vectorutil.Cosine(queryVector, vector.Values)
		if err != nil {
			return nil, fmt.Errorf("score vector %q: %w", vector.RepresentationID, err)
		}
		representation := index.representations[vector.RepresentationID]
		chunk := index.chunks[representation.ChunkID]
		hits = append(hits, rag.Hit{
			RepresentationID: representation.ID,
			ChunkID:          chunk.ID,
			DocumentID:       chunk.DocumentID,
			Channel:          index.channel,
			Score:            score,
		})
	}
	sort.Slice(hits, func(left, right int) bool {
		return rag.HitRanksBefore(hits[left], hits[right])
	})
	if len(hits) > limit {
		hits = hits[:limit]
	}
	for position := range hits {
		hits[position].Rank = position + 1
	}
	return hits, nil
}

func (index *Exact) Close() error { return nil }
