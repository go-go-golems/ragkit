package sqliteexact

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
)

type fakeEmbedder struct {
	vector []float32
}

func (f fakeEmbedder) Embed(context.Context, rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	return rag.EmbeddingResult{Vectors: [][]float32{f.vector}}, nil
}

func TestBuildSearchReopen(t *testing.T) {
	t.Parallel()
	representations := []rag.Representation{
		{ID: "rep-a", ChunkID: "chunk-a", ContentDigest: "a"},
		{ID: "rep-b", ChunkID: "chunk-b", ContentDigest: "b"},
	}
	chunks := []rag.Chunk{
		{ID: "chunk-a", DocumentID: "doc-a"},
		{ID: "chunk-b", DocumentID: "doc-b"},
	}
	vectors := []rag.Vector{
		{RepresentationID: "rep-a", Model: "model", Values: []float32{1, 0}},
		{RepresentationID: "rep-b", Model: "model", Values: []float32{0, 1}},
	}
	path := filepath.Join(t.TempDir(), "vectors.sqlite")
	index, err := Build(context.Background(), Config{Path: path, Model: "model"}, representations, chunks, vectors, fakeEmbedder{vector: []float32{1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	hits, err := index.Search(context.Background(), rag.Query{Text: "a"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].RepresentationID != "rep-a" || hits[0].Rank != 1 {
		t.Fatalf("unexpected hits: %#v", hits)
	}
	if err := index.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := Open(path, "model", "", fakeEmbedder{vector: []float32{1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = reopened.Close() }()
	hits, err = reopened.Search(context.Background(), rag.Query{Text: "a"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 || hits[0].RepresentationID != "rep-a" {
		t.Fatalf("unexpected reopened hits: %#v", hits)
	}
}

func TestSearchUsesChunkIDAtEqualScoreCutoff(t *testing.T) {
	t.Parallel()
	representations := []rag.Representation{
		{ID: "rep-a", ChunkID: "chunk-z", ContentDigest: "a"},
		{ID: "rep-z", ChunkID: "chunk-a", ContentDigest: "z"},
	}
	chunks := []rag.Chunk{
		{ID: "chunk-z", DocumentID: "doc-z"},
		{ID: "chunk-a", DocumentID: "doc-a"},
	}
	vectors := []rag.Vector{
		{RepresentationID: "rep-a", Model: "model", Values: []float32{1, 0}},
		{RepresentationID: "rep-z", Model: "model", Values: []float32{1, 0}},
	}
	path := filepath.Join(t.TempDir(), "tie-vectors.sqlite")
	index, err := Build(context.Background(), Config{Path: path, Model: "model"}, representations, chunks, vectors, fakeEmbedder{vector: []float32{1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = index.Close() }()
	hits, err := index.Search(context.Background(), rag.Query{Text: "query"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].ChunkID != "chunk-a" {
		t.Fatalf("unexpected tied cutoff: %#v", hits)
	}
}

func TestReadEntriesUsesStableRepresentationOrder(t *testing.T) {
	t.Parallel()
	representations := []rag.Representation{
		{ID: "rep-z", ChunkID: "chunk-z", ContentDigest: "z"},
		{ID: "rep-a", ChunkID: "chunk-a", ContentDigest: "a"},
	}
	chunks := []rag.Chunk{
		{ID: "chunk-z", DocumentID: "doc-z"},
		{ID: "chunk-a", DocumentID: "doc-a"},
	}
	vectors := []rag.Vector{
		{RepresentationID: "rep-z", Model: "model", Values: []float32{0, 1}},
		{RepresentationID: "rep-a", Model: "model", Values: []float32{1, 0}},
	}
	path := filepath.Join(t.TempDir(), "entries.sqlite")
	index, err := Build(context.Background(), Config{Path: path, Model: "model"}, representations, chunks, vectors, fakeEmbedder{vector: []float32{1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	if err := index.Close(); err != nil {
		t.Fatal(err)
	}
	entries, manifest, err := ReadEntries(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.RepresentationCount != 2 || len(entries) != 2 {
		t.Fatalf("manifest/entries = %+v/%d", manifest, len(entries))
	}
	if entries[0].RepresentationID != "rep-a" || entries[1].RepresentationID != "rep-z" {
		t.Fatalf("entries are not stably ordered: %#v", entries)
	}
	if entries[0].ChunkID != "chunk-a" || entries[0].DocumentID != "doc-a" || entries[0].Values[0] != 1 {
		t.Fatalf("unexpected first entry: %#v", entries[0])
	}
}
