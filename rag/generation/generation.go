package generation

import (
	"context"
	"fmt"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
)

// Raw turns source chunks into raw searchable representations.
func Raw(chunks []rag.Chunk) []rag.Representation {
	result := make([]rag.Representation, len(chunks))
	for index, chunk := range chunks {
		result[index] = rag.Representation{
			ID:            "rep-raw-" + chunk.ID,
			ChunkID:       chunk.ID,
			Kind:          "raw",
			Text:          chunk.Text,
			ContentDigest: chunk.ContentDigest,
		}
	}
	return result
}

// Extractive is a deterministic answer generator for examples and tests.
type Extractive struct {
	MaxEvidence int
}

var _ rag.Generator = (*Extractive)(nil)

func (generator *Extractive) Generate(ctx context.Context, request rag.GenerationRequest) (rag.GenerationResult, error) {
	if err := ctx.Err(); err != nil {
		return rag.GenerationResult{}, err
	}
	if len(request.Evidence) == 0 {
		return rag.GenerationResult{Text: "I do not have enough evidence.", FinishReason: "abstain"}, nil
	}
	limit := generator.MaxEvidence
	if limit < 1 || limit > len(request.Evidence) {
		limit = len(request.Evidence)
	}
	var builder strings.Builder
	citations := make([]string, 0, limit)
	for index, evidence := range request.Evidence[:limit] {
		if index > 0 {
			builder.WriteString("\n\n")
		}
		fmt.Fprintf(&builder, "[%s] %s", evidence.Chunk.ID, strings.TrimSpace(evidence.Chunk.Text))
		citations = append(citations, evidence.Chunk.ID)
	}
	output := builder.String()
	inputTokens := int64(len(request.Text) / 4)
	outputTokens := int64(len(output) / 4)
	return rag.GenerationResult{
		Text:         output,
		Citations:    citations,
		FinishReason: "stop",
		Usage: rag.Usage{
			InputTokens:  &inputTokens,
			OutputTokens: &outputTokens,
		},
	}, nil
}
