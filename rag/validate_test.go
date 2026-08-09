package rag

import (
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/stretchr/testify/require"
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

func TestValidateCorpusRejectsInvalidOrdinals(t *testing.T) {
	document := Document{ID: "doc", Text: "ab", ContentDigest: digest.Text("ab")}
	chunk := func(id string, ordinal, start int) Chunk {
		text := document.Text[start : start+1]
		return Chunk{ID: id, DocumentID: document.ID, Ordinal: ordinal, Range: Range{ByteStart: start, ByteEnd: start + 1}, Text: text, ContentDigest: digest.Text(text), Chunker: "test"}
	}
	require.ErrorContains(t, ValidateCorpus([]Document{document}, []Chunk{chunk("negative", -1, 0)}), "negative ordinal")
	require.ErrorContains(t, ValidateCorpus([]Document{document}, []Chunk{chunk("a", 0, 0), chunk("b", 0, 1)}), "duplicates ordinal")
}
