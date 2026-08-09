package chunking

import (
	"context"
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
)

func TestFixedPreservesUnicodeSourceRanges(t *testing.T) {
	t.Parallel()
	document := rag.Document{ID: "doc", Text: "one 🌳 two three", ContentDigest: digest.Text("one 🌳 two three")}
	chunks, err := (&Fixed{SizeRunes: 6, OverlapRunes: 1}).Chunk(context.Background(), document)
	if err != nil {
		t.Fatalf("Chunk() error = %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("chunks = %d, want >= 2", len(chunks))
	}
	const wantFirstID = "chunk-51d9d498f37a5843"
	if chunks[0].ID != wantFirstID {
		t.Fatalf("first chunk ID = %q, want %q", chunks[0].ID, wantFirstID)
	}
	const wantFirstDigest = "d71459893569ba3bd6f25f40f935a8aa6ffef216147c6eb3654c9028f7b1e226"
	if chunks[0].ContentDigest != wantFirstDigest {
		t.Fatalf("first chunk content digest = %q, want %q", chunks[0].ContentDigest, wantFirstDigest)
	}
	for _, chunk := range chunks {
		if err := rag.ValidateChunk(document, chunk); err != nil {
			t.Fatalf("ValidateChunk(%s) error = %v", chunk.ID, err)
		}
	}
}

func TestMarkdownStartsChunksAtHeadings(t *testing.T) {
	t.Parallel()
	text := "# First\nalpha\n## Second\nbeta\n"
	document := rag.Document{ID: "doc", Text: text, ContentDigest: digest.Text(text)}
	chunks, err := (&Markdown{MaxSectionRunes: 100, OverlapRunes: 0}).Chunk(context.Background(), document)
	if err != nil {
		t.Fatalf("Chunk() error = %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2", len(chunks))
	}
	if chunks[1].Text != "## Second\nbeta\n" {
		t.Fatalf("second chunk = %q", chunks[1].Text)
	}
}

func TestMarkdownOnlySplitsAtStructuralATXHeadings(t *testing.T) {
	t.Parallel()
	text := "intro\n#hashtag\n    # indented code\n```go\n# fenced code\n```\n~~~\n## fenced too\n~~~\n   ## Real\nbody\n####### not a heading\n"
	document := rag.Document{ID: "doc", Text: text, ContentDigest: digest.Text(text)}
	chunks, err := (&Markdown{MaxSectionRunes: 1_000, OverlapRunes: 0}).Chunk(context.Background(), document)
	if err != nil {
		t.Fatalf("Chunk() error = %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2: %+v", len(chunks), chunks)
	}
	if chunks[1].Text != "   ## Real\nbody\n####### not a heading\n" {
		t.Fatalf("second chunk = %q", chunks[1].Text)
	}
}
