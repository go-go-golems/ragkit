package vector

import (
	"context"
	"math"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/go-go-golems/ragkit/rag/embedding"
	"github.com/go-go-golems/ragkit/rag/generation"
)

type fixedEmbedder struct{ vector []float32 }

func (f fixedEmbedder) Embed(context.Context, rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	return rag.EmbeddingResult{Vectors: [][]float32{f.vector}}, nil
}

func TestExactSearch(t *testing.T) {
	t.Parallel()
	chunks := []rag.Chunk{{ID: "oak", DocumentID: "doc-oak", Text: "oak soil"}, {ID: "pine", DocumentID: "doc-pine", Text: "pine needles"}}
	reps := generation.Raw(chunks)
	embedder := &embedding.HashEmbedder{Dimensions: 32}
	vectors, _, err := embedding.Representations(context.Background(), embedder, "hash", reps, 2)
	if err != nil {
		t.Fatalf("Representations() error = %v", err)
	}
	index, err := NewExact("hash", "vector", embedder, reps, chunks, vectors)
	if err != nil {
		t.Fatalf("NewExact() error = %v", err)
	}
	hits, err := index.Search(context.Background(), rag.Query{Text: "oak soil"}, 1)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(hits) != 1 || hits[0].ChunkID != "oak" {
		t.Fatalf("hits = %+v", hits)
	}
}

func TestNewExactRejectsMismatchedVectorModel(t *testing.T) {
	t.Parallel()
	chunks := []rag.Chunk{{ID: "chunk", DocumentID: "doc"}}
	reps := generation.Raw(chunks)
	vectors := []rag.Vector{{RepresentationID: reps[0].ID, Model: "other", Values: []float32{1}}}
	if _, err := NewExact("wanted", "vector", &embedding.HashEmbedder{Dimensions: 1}, reps, chunks, vectors); err == nil {
		t.Fatal("NewExact() error = nil, want model mismatch")
	}
}

func TestNewExactRejectsNonFiniteVectors(t *testing.T) {
	t.Parallel()
	chunks := []rag.Chunk{{ID: "chunk", DocumentID: "doc"}}
	reps := generation.Raw(chunks)
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		vectors := []rag.Vector{{RepresentationID: reps[0].ID, Model: "model", Values: []float32{value}}}
		if _, err := NewExact("model", "vector", &embedding.HashEmbedder{Dimensions: 1}, reps, chunks, vectors); err == nil {
			t.Fatalf("NewExact() error = nil for value %v", value)
		}
	}
}

func TestExactSearchUsesChunkIDForEqualScores(t *testing.T) {
	t.Parallel()
	chunks := []rag.Chunk{
		{ID: "chunk-z", DocumentID: "doc-z"},
		{ID: "chunk-a", DocumentID: "doc-a"},
	}
	representations := []rag.Representation{
		{ID: "rep-a", ChunkID: "chunk-z"},
		{ID: "rep-z", ChunkID: "chunk-a"},
	}
	vectors := []rag.Vector{
		{RepresentationID: "rep-a", Model: "model", Values: []float32{1, 0}},
		{RepresentationID: "rep-z", Model: "model", Values: []float32{1, 0}},
	}
	index, err := NewExact("model", "vector", fixedEmbedder{vector: []float32{1, 0}}, representations, chunks, vectors)
	if err != nil {
		t.Fatal(err)
	}
	hits, err := index.Search(context.Background(), rag.Query{Text: "query"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if hits[0].ChunkID != "chunk-a" {
		t.Fatalf("first tied chunk = %q, want chunk-a", hits[0].ChunkID)
	}
}
