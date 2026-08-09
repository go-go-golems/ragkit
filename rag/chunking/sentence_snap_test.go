package chunking

import (
	"context"
	"strings"
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
)

func snapDocument(text string) rag.Document {
	return rag.Document{ID: "doc-1", Title: "T", Text: text, ContentDigest: digest.Text(text)}
}

func TestSentenceSnapEndsChunksAtSentenceBoundaries(t *testing.T) {
	// Sentences of 40 runes; a 100-rune window without snapping cuts
	// mid-sentence, with snapping it must end on a period.
	sentence := strings.Repeat("x", 39) + "."
	document := snapDocument(strings.Repeat(sentence, 10))
	chunker := &Markdown{MaxSectionRunes: 100, OverlapRunes: 10, SentenceSnapRunes: 50}
	chunks, err := chunker.Chunk(context.Background(), document)
	if err != nil {
		t.Fatal(err)
	}
	for i, chunk := range chunks[:len(chunks)-1] {
		if !strings.HasSuffix(chunk.Text, ".") {
			t.Fatalf("chunk %d does not end at a sentence boundary: ...%q", i, chunk.Text[len(chunk.Text)-10:])
		}
	}
}

func TestSentenceSnapWithoutBoundaryKeepsTheWindow(t *testing.T) {
	document := snapDocument(strings.Repeat("y", 300))
	chunker := &Markdown{MaxSectionRunes: 100, OverlapRunes: 10, SentenceSnapRunes: 50}
	chunks, err := chunker.Chunk(context.Background(), document)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks[0].Text) != 100 {
		t.Fatalf("boundary-free text must keep the full window, got %d runes", len(chunks[0].Text))
	}
}

func TestSentenceSnapChangesTheChunkerName(t *testing.T) {
	plain := &Markdown{MaxSectionRunes: 1200, OverlapRunes: 120}
	snapped := &Markdown{MaxSectionRunes: 1200, OverlapRunes: 120, SentenceSnapRunes: 200}
	if plain.Name() == snapped.Name() {
		t.Fatal("snap parameter must enter the chunker identity")
	}
	if snapped.Name() != "markdown-1200-120-snap200" {
		t.Fatalf("name = %q", snapped.Name())
	}
}

func TestSentenceSnapStillTerminatesOnPathologicalText(t *testing.T) {
	// Periods everywhere: snapping must never stall the window.
	document := snapDocument(strings.Repeat(".", 500))
	chunker := &Markdown{MaxSectionRunes: 100, OverlapRunes: 90, SentenceSnapRunes: 99}
	chunks, err := chunker.Chunk(context.Background(), document)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Fatal("no chunks")
	}
}
