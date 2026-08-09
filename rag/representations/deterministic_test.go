package representations

import (
	"strings"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
)

func TestBreadcrumbsPrependTheHeadingPath(t *testing.T) {
	documents, chunks := labChunks()
	reps, err := Breadcrumbs(documents, chunks)
	if err != nil {
		t.Fatal(err)
	}
	if len(reps) != 1 || reps[0].Kind != KindBreadcrumb {
		t.Fatalf("reps = %+v", reps)
	}
	want := "Thuja Green Giant > Planting > Spacing\nPlant 5 feet apart."
	if reps[0].Text != want {
		t.Fatalf("breadcrumb text = %q, want %q", reps[0].Text, want)
	}
	if reps[0].ChunkID != chunks[0].ID {
		t.Fatalf("breadcrumb must hydrate to its source chunk")
	}
}

func TestSmallToBigIndexesSmallTextUnderTheParent(t *testing.T) {
	parent := rag.Chunk{
		ID: "parent-1", DocumentID: "doc-1",
		Range: rag.Range{ByteStart: 0, ByteEnd: 100},
		Text:  strings.Repeat("p", 100), ContentDigest: "pd",
	}
	small := rag.Chunk{
		ID: "small-1", DocumentID: "doc-1",
		Range: rag.Range{ByteStart: 10, ByteEnd: 40},
		Text:  "precise small text", ContentDigest: "sd",
	}
	reps, err := SmallToBig([]rag.Chunk{small}, []rag.Chunk{parent})
	if err != nil {
		t.Fatal(err)
	}
	if reps[0].ChunkID != "parent-1" {
		t.Fatalf("small rep must point at the parent, got %q", reps[0].ChunkID)
	}
	if reps[0].Text != "precise small text" || reps[0].Kind != KindSmall {
		t.Fatalf("rep = %+v", reps[0])
	}
}

func TestSmallToBigPicksTheLargestOverlap(t *testing.T) {
	parentA := rag.Chunk{ID: "a", DocumentID: "d", Range: rag.Range{ByteStart: 0, ByteEnd: 50}, Text: "a", ContentDigest: "a"}
	parentB := rag.Chunk{ID: "b", DocumentID: "d", Range: rag.Range{ByteStart: 40, ByteEnd: 120}, Text: "b", ContentDigest: "b"}
	small := rag.Chunk{ID: "s", DocumentID: "d", Range: rag.Range{ByteStart: 45, ByteEnd: 75}, Text: "s", ContentDigest: "s"}
	reps, err := SmallToBig([]rag.Chunk{small}, []rag.Chunk{parentA, parentB})
	if err != nil {
		t.Fatal(err)
	}
	if reps[0].ChunkID != "b" {
		t.Fatalf("overlap 30 with b beats 5 with a; got %q", reps[0].ChunkID)
	}
}

func TestSmallToBigFailsWithoutACoveringParent(t *testing.T) {
	small := rag.Chunk{ID: "s", DocumentID: "d", Range: rag.Range{ByteStart: 0, ByteEnd: 10}, Text: "s", ContentDigest: "s"}
	if _, err := SmallToBig([]rag.Chunk{small}, nil); err == nil {
		t.Fatal("orphan small chunk must fail, not silently drop")
	}
}

func TestSmallToBigDeduplicatesIdenticalRepresentations(t *testing.T) {
	parent := rag.Chunk{ID: "parent", DocumentID: "d", Range: rag.Range{ByteStart: 0, ByteEnd: 20}, Text: "parent", ContentDigest: "p"}
	small := []rag.Chunk{
		{ID: "small-a", DocumentID: "d", Range: rag.Range{ByteStart: 0, ByteEnd: 4}, Text: "same", ContentDigest: "a"},
		{ID: "small-b", DocumentID: "d", Range: rag.Range{ByteStart: 5, ByteEnd: 9}, Text: "same", ContentDigest: "b"},
	}
	reps, err := SmallToBig(small, []rag.Chunk{parent})
	if err != nil {
		t.Fatal(err)
	}
	if len(reps) != 1 {
		t.Fatalf("representations = %d, want one unique identity", len(reps))
	}
}
