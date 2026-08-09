package rag

import (
	"testing"

	"github.com/go-go-golems/ragkit/digest"
)

func TestValidateChunk(t *testing.T) {
	t.Parallel()

	document := Document{
		ID:            "doc-1",
		Text:          "hello 🌳 world",
		ContentDigest: digest.Text("hello 🌳 world"),
	}
	chunk := Chunk{
		ID:            "chunk-1",
		DocumentID:    document.ID,
		Range:         Range{ByteStart: 6, ByteEnd: 10},
		Text:          "🌳",
		ContentDigest: digest.Text("🌳"),
		Chunker:       "fixed",
	}

	if err := ValidateChunk(document, chunk); err != nil {
		t.Fatalf("ValidateChunk() error = %v", err)
	}
}

func TestValidateChunkRejectsMismatchedSourceSlice(t *testing.T) {
	t.Parallel()

	document := Document{ID: "doc-1", Text: "source", ContentDigest: digest.Text("source")}
	chunk := Chunk{
		ID:            "chunk-1",
		DocumentID:    document.ID,
		Range:         Range{ByteStart: 0, ByteEnd: 6},
		Text:          "target",
		ContentDigest: digest.Text("target"),
		Chunker:       "fixed",
	}

	if err := ValidateChunk(document, chunk); err == nil {
		t.Fatal("ValidateChunk() error = nil, want source mismatch")
	}
}
