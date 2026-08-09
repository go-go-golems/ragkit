package chunking

import (
	"fmt"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
)

// FromRanges builds validated chunks from explicit byte ranges. It is the
// entry point for chunkers whose boundaries come from outside the package —
// the llm-chunk experiment proposes cut points with a model and aligns them
// to byte offsets, but the chunks it publishes must satisfy the same
// exact-slice invariant as every window chunk. Ranges must be in order,
// non-overlapping, and cover only real document bytes; ordinals follow slice
// order.
func FromRanges(document rag.Document, name string, ranges []rag.Range) ([]rag.Chunk, error) {
	if name == "" {
		return nil, fmt.Errorf("chunker name is required")
	}
	chunks := make([]rag.Chunk, 0, len(ranges))
	previousEnd := 0
	for index, byteRange := range ranges {
		if byteRange.ByteStart < previousEnd || byteRange.ByteEnd <= byteRange.ByteStart ||
			byteRange.ByteEnd > len(document.Text) {
			return nil, fmt.Errorf(
				"range %d [%d,%d) is out of order or outside document %s (%d bytes)",
				index, byteRange.ByteStart, byteRange.ByteEnd, document.ID, len(document.Text),
			)
		}
		previousEnd = byteRange.ByteEnd
		text := document.Text[byteRange.ByteStart:byteRange.ByteEnd]
		idDigest, err := digest.JSON(struct {
			DocumentID string `json:"document_id"`
			Chunker    string `json:"chunker"`
			ByteStart  int    `json:"byte_start"`
			ByteEnd    int    `json:"byte_end"`
			TextDigest string `json:"text_digest"`
		}{
			DocumentID: document.ID,
			Chunker:    name,
			ByteStart:  byteRange.ByteStart,
			ByteEnd:    byteRange.ByteEnd,
			TextDigest: digest.Text(text),
		})
		if err != nil {
			return nil, err
		}
		chunk := rag.Chunk{
			ID:            "chunk-" + idDigest[:16],
			DocumentID:    document.ID,
			Ordinal:       index,
			Range:         byteRange,
			Text:          text,
			ContentDigest: digest.Text(text),
			Chunker:       name,
		}
		if err := rag.ValidateChunk(document, chunk); err != nil {
			return nil, err
		}
		chunks = append(chunks, chunk)
	}
	return chunks, nil
}
