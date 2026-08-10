package geppetto

import (
	"context"
	"fmt"
	"strings"

	"github.com/go-go-golems/geppetto/pkg/embeddings"
	"github.com/go-go-golems/ragkit/rag"
	vectorutil "github.com/go-go-golems/ragkit/vector"
)

// Embedder adapts one already-configured Geppetto embeddings provider. Provider
// construction and input task prefixes remain application responsibilities.
type Embedder struct {
	provider embeddings.Provider
	model    embeddings.EmbeddingModel
}

var _ rag.Embedder = (*Embedder)(nil)

// NewEmbedder validates and wraps a configured Geppetto embeddings provider.
func NewEmbedder(provider embeddings.Provider) (*Embedder, error) {
	if provider == nil {
		return nil, fmt.Errorf("embedding provider is required")
	}
	model := provider.GetModel()
	if strings.TrimSpace(model.Name) == "" || model.Dimensions < 1 {
		return nil, fmt.Errorf("embedding provider has invalid model metadata")
	}
	return &Embedder{provider: provider, model: model}, nil
}

// Embed translates one ragkit embedding batch and validates the complete
// provider response before exposing vectors to an index or cache.
func (e *Embedder) Embed(ctx context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	if e == nil || e.provider == nil {
		return rag.EmbeddingResult{}, fmt.Errorf("embedding provider is unavailable")
	}
	if request.Model != "" && request.Model != e.model.Name {
		return rag.EmbeddingResult{}, fmt.Errorf(
			"embedding model mismatch: requested %q, provider %q",
			request.Model,
			e.model.Name,
		)
	}
	if len(request.Texts) == 0 {
		return rag.EmbeddingResult{}, fmt.Errorf("embedding texts are required")
	}

	vectors, err := e.provider.GenerateBatchEmbeddings(ctx, request.Texts)
	if err != nil {
		return rag.EmbeddingResult{}, fmt.Errorf("generate Geppetto embeddings: %w", err)
	}
	if len(vectors) != len(request.Texts) {
		return rag.EmbeddingResult{}, fmt.Errorf(
			"embedding result count mismatch: got %d, want %d",
			len(vectors),
			len(request.Texts),
		)
	}
	for i, vector := range vectors {
		if len(vector) != e.model.Dimensions {
			return rag.EmbeddingResult{}, fmt.Errorf(
				"embedding %d dimensions: got %d, want %d",
				i,
				len(vector),
				e.model.Dimensions,
			)
		}
		if err := vectorutil.ValidateFinite(vector); err != nil {
			return rag.EmbeddingResult{}, fmt.Errorf("embedding %d: %w", i, err)
		}
	}

	return rag.EmbeddingResult{Vectors: vectors}, nil
}
