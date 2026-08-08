package chunking

import (
	"context"
	"strings"
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
)

// TestMarkdownHeadingKeepsSectionsWhole verifies the core claim of the
// chunker: when a section fits the maximum, it is emitted as one chunk rather
// than re-windowed. The existing Markdown chunker would cut this into two
// 1200-rune pieces; MarkdownHeading keeps each section intact.
func TestMarkdownHeadingKeepsSectionsWhole(t *testing.T) {
	t.Parallel()
	text := "# First\n" + strings.Repeat("a", 300) + "\n" +
		"# Second\n" + strings.Repeat("b", 300) + "\n"
	document := rag.Document{
		ID: "doc", Text: text, ContentDigest: digest.Text(text),
	}
	chunks, err := (&MarkdownHeading{
		MaxSectionRunes: 1200, MinSectionRunes: 100, OverlapRunes: 120,
	}).Chunk(context.Background(), document)
	if err != nil {
		t.Fatalf("Chunk() error = %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2 (one per section)", len(chunks))
	}
	for _, chunk := range chunks {
		if err := rag.ValidateChunk(document, chunk); err != nil {
			t.Fatalf("ValidateChunk(%s) error = %v", chunk.ID, err)
		}
	}
	if !strings.HasPrefix(chunks[0].Text, "# First") {
		t.Fatalf("first chunk does not start at its heading: %q", chunks[0].Text[:10])
	}
	if !strings.HasPrefix(chunks[1].Text, "# Second") {
		t.Fatalf("second chunk does not start at its heading: %q", chunks[1].Text[:10])
	}
}

// TestMarkdownHeadingSplitsOversizedSection verifies that a section larger than
// the maximum falls back to a fixed window with overlap, so a giant section is
// still retrievable in pieces.
func TestMarkdownHeadingSplitsOversizedSection(t *testing.T) {
	t.Parallel()
	text := "# Big\n" + strings.Repeat("x", 3000) + "\n"
	document := rag.Document{
		ID: "doc", Text: text, ContentDigest: digest.Text(text),
	}
	chunks, err := (&MarkdownHeading{
		MaxSectionRunes: 1200, MinSectionRunes: 100, OverlapRunes: 120,
	}).Chunk(context.Background(), document)
	if err != nil {
		t.Fatalf("Chunk() error = %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("chunks = %d, want >= 2 for an oversized section", len(chunks))
	}
	// The first chunk must start at the heading; the window fallback preserves
	// the section base.
	if !strings.HasPrefix(chunks[0].Text, "# Big") {
		t.Fatalf("first chunk does not start at its heading: %q", chunks[0].Text[:10])
	}
	for _, chunk := range chunks {
		if err := rag.ValidateChunk(document, chunk); err != nil {
			t.Fatalf("ValidateChunk(%s) error = %v", chunk.ID, err)
		}
	}
}

// TestMarkdownHeadingMergesSmallSections verifies that a tiny section is merged
// into its successor rather than emitted as a shard. This is the fix for the
// "Arkansas Trees" heading-cuts the assessment found.
func TestMarkdownHeadingMergesSmallSections(t *testing.T) {
	t.Parallel()
	text := "#1. Rainbow Eucalyptus\nIdeal for providing shade.\n" +
		"#2. Tulip Poplar\n" + strings.Repeat("y", 400) + "\n"
	document := rag.Document{
		ID: "doc", Text: text, ContentDigest: digest.Text(text),
	}
	chunks, err := (&MarkdownHeading{
		MaxSectionRunes: 1200, MinSectionRunes: 200, OverlapRunes: 120,
	}).Chunk(context.Background(), document)
	if err != nil {
		t.Fatalf("Chunk() error = %v", err)
	}
	// The tiny first section is merged into the second, producing one chunk.
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1 (small section merged)", len(chunks))
	}
	if err := rag.ValidateChunk(document, chunks[0]); err != nil {
		t.Fatalf("ValidateChunk error = %v", err)
	}
	// The merged chunk must contain both headings' text.
	if !strings.Contains(chunks[0].Text, "Rainbow Eucalyptus") ||
		!strings.Contains(chunks[0].Text, "Tulip Poplar") {
		t.Fatalf("merged chunk lost a section: %q", chunks[0].Text[:40])
	}
}

func TestMarkdownHeadingMergesInteriorSmallSectionForward(t *testing.T) {
	t.Parallel()
	first := "# First\n" + strings.Repeat("a", 250) + "\n"
	tiny := "# Tiny\nshort\n"
	third := "# Third\n" + strings.Repeat("b", 250) + "\n"
	text := first + tiny + third
	document := rag.Document{ID: "doc", Text: text, ContentDigest: digest.Text(text)}

	chunks, err := (&MarkdownHeading{
		MaxSectionRunes: 1200, MinSectionRunes: 100, OverlapRunes: 120,
	}).Chunk(context.Background(), document)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2", len(chunks))
	}
	if strings.Contains(chunks[0].Text, "# Tiny") || !strings.HasPrefix(chunks[1].Text, "# Tiny") || !strings.Contains(chunks[1].Text, "# Third") {
		t.Fatalf("interior tiny section did not merge forward: %#v", chunks)
	}
}

// TestMarkdownHeadingNameIncludesParameters guards the stable identity used in
// bundle manifests and reports.
func TestMarkdownHeadingNameIncludesParameters(t *testing.T) {
	t.Parallel()
	name := (&MarkdownHeading{
		MaxSectionRunes: 1200, MinSectionRunes: 200, OverlapRunes: 120,
	}).Name()
	const want = "markdown-heading-1200-200-120"
	if name != want {
		t.Fatalf("Name() = %q, want %q", name, want)
	}
}
