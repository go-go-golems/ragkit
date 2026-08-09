package embedding

import (
	"context"
	"fmt"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	textutil "github.com/go-go-golems/ragkit/text"
	vectorutil "github.com/go-go-golems/ragkit/vector"
)

// HashEmbedder maps lowercase word tokens into a signed hashing vector. It is
// deterministic and useful for tests and examples, not a semantic model.
type HashEmbedder struct {
	Dimensions int
}

var _ rag.Embedder = (*HashEmbedder)(nil)

func (embedder *HashEmbedder) Embed(ctx context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	if embedder == nil || embedder.Dimensions < 1 {
		return rag.EmbeddingResult{}, fmt.Errorf("embedding dimensions must be positive")
	}
	vectors := make([][]float32, len(request.Texts))
	var tokenCount int64
	for index, text := range request.Texts {
		if err := ctx.Err(); err != nil {
			return rag.EmbeddingResult{}, err
		}
		vector := make([]float32, embedder.Dimensions)
		for _, token := range textutil.Terms(text) {
			tokenCount++
			hash := digest.Text(token)
			position := 0
			for offset := 0; offset < 16; offset++ {
				position = (position*16 + int(hexValue(hash[offset]))) % embedder.Dimensions
			}
			sign := float32(1)
			if hexValue(hash[15])&1 == 1 {
				sign = -1
			}
			vector[position] += sign
		}
		if err := vectorutil.Normalize(vector); err != nil {
			return rag.EmbeddingResult{}, fmt.Errorf("normalize hash embedding: %w", err)
		}
		vectors[index] = vector
	}
	return rag.EmbeddingResult{
		Vectors: vectors,
		Usage:   rag.Usage{EmbeddingTokens: &tokenCount},
	}, nil
}

func hexValue(value byte) byte {
	if value >= '0' && value <= '9' {
		return value - '0'
	}
	return value - 'a' + 10
}
