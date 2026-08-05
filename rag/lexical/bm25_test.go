package lexical

import (
	"context"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/go-go-golems/ragkit/rag/generation"
)

func TestBM25FindsMatchingChunk(t *testing.T) {
	t.Parallel()
	chunks := []rag.Chunk{
		{ID: "oak", DocumentID: "doc-oak", Text: "oak trees prefer well drained soil"},
		{ID: "pine", DocumentID: "doc-pine", Text: "pine needles stay green"},
	}
	index, err := Build(generation.Raw(chunks), chunks, Config{})
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	hits, err := index.Search(context.Background(), rag.Query{Text: "oak soil"}, 2)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) == 0 || hits[0].ChunkID != "oak" {
		t.Fatalf("hits = %+v", hits)
	}
}

func TestBM25PreservesExplicitZeroB(t *testing.T) {
	t.Parallel()
	zero := 0.0
	chunks := []rag.Chunk{
		{ID: "short", DocumentID: "short", Text: "oak"},
		{ID: "long", DocumentID: "long", Text: "oak filler filler filler"},
	}
	index, err := Build(generation.Raw(chunks), chunks, Config{B: &zero})
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	hits, err := index.Search(context.Background(), rag.Query{Text: "oak"}, 2)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if hits[0].Score != hits[1].Score {
		t.Fatalf("scores = %v and %v, want equal scores with B=0", hits[0].Score, hits[1].Score)
	}
	if hits[0].ChunkID != "long" {
		t.Fatalf("first tied chunk = %q, want chunk-ID order starting with long", hits[0].ChunkID)
	}
}
