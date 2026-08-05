package rag

import (
	"fmt"
	"unicode/utf8"
)

// ValidateDocument checks the invariants common to every corpus loader.
func ValidateDocument(document Document) error {
	if document.ID == "" {
		return fmt.Errorf("document ID is required")
	}
	if !utf8.ValidString(document.Text) {
		return fmt.Errorf("document %q contains invalid UTF-8", document.ID)
	}
	if document.ContentDigest == "" {
		return fmt.Errorf("document %q content digest is required", document.ID)
	}
	return nil
}

// ValidateChunk checks identity, source range, and exact source-slice equality.
func ValidateChunk(document Document, chunk Chunk) error {
	if err := ValidateDocument(document); err != nil {
		return err
	}
	if chunk.ID == "" {
		return fmt.Errorf("chunk ID is required")
	}
	if chunk.DocumentID != document.ID {
		return fmt.Errorf("chunk %q belongs to document %q, not %q", chunk.ID, chunk.DocumentID, document.ID)
	}
	if chunk.Range.ByteStart < 0 || chunk.Range.ByteEnd < chunk.Range.ByteStart || chunk.Range.ByteEnd > len(document.Text) {
		return fmt.Errorf("chunk %q has invalid byte range [%d,%d)", chunk.ID, chunk.Range.ByteStart, chunk.Range.ByteEnd)
	}
	if document.Text[chunk.Range.ByteStart:chunk.Range.ByteEnd] != chunk.Text {
		return fmt.Errorf("chunk %q text does not match its source range", chunk.ID)
	}
	if !utf8.ValidString(chunk.Text) {
		return fmt.Errorf("chunk %q contains invalid UTF-8", chunk.ID)
	}
	if chunk.ContentDigest == "" {
		return fmt.Errorf("chunk %q content digest is required", chunk.ID)
	}
	if chunk.Chunker == "" {
		return fmt.Errorf("chunk %q chunker is required", chunk.ID)
	}
	return nil
}
