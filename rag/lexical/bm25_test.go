package lexical

import (
	"context"
	"math"
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

func TestBM25RejectsNonFiniteParameters(t *testing.T) {
	t.Parallel()
	chunks := []rag.Chunk{{ID: "chunk", DocumentID: "doc", Text: "oak"}}
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := Build(generation.Raw(chunks), chunks, Config{K1: value}); err == nil {
			t.Fatalf("Build(K1=%v) error = nil", value)
		}
		if _, err := Build(generation.Raw(chunks), chunks, Config{K1: 1.2, B: &value}); err == nil {
			t.Fatalf("Build(B=%v) error = nil", value)
		}
	}
}

func TestBM25RejectsDuplicateRepresentationIDs(t *testing.T) {
	chunks := []rag.Chunk{{ID: "a", DocumentID: "doc", Text: "oak"}, {ID: "b", DocumentID: "doc", Text: "pine"}}
	representations := []rag.Representation{{ID: "duplicate", ChunkID: "a", Text: "oak"}, {ID: "duplicate", ChunkID: "b", Text: "pine"}}
	if _, err := Build(representations, chunks, Config{}); err == nil {
		t.Fatal("Build() accepted duplicate representation IDs")
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

func TestBM25OwnsConfigurationValues(t *testing.T) {
	t.Parallel()
	b := 0.0
	chunks := []rag.Chunk{
		{ID: "short", DocumentID: "short", Text: "oak"},
		{ID: "long", DocumentID: "long", Text: "oak filler filler filler"},
	}
	index, err := Build(generation.Raw(chunks), chunks, Config{B: &b})
	if err != nil {
		t.Fatal(err)
	}
	b = 1
	hits, err := index.Search(context.Background(), rag.Query{Text: "oak"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if hits[0].Score != hits[1].Score {
		t.Fatalf("caller mutation changed stored B: scores = %v and %v", hits[0].Score, hits[1].Score)
	}
}
