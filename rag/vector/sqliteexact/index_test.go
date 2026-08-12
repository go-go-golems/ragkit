package sqliteexact

import (
	"context"
	"encoding/binary"
	"math"
	"net/url"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	vectorutil "github.com/go-go-golems/ragkit/vector"
)

func TestDecodeRejectsOverflowingDimensions(t *testing.T) {
	if _, err := decode([]byte{0, 0, 0, 0}, math.MaxInt); err == nil {
		t.Fatal("decode accepted dimensions whose byte-size multiplication would overflow")
	}
}

func TestCosineBlobMatchesVectorCosineWithoutDecoding(t *testing.T) {
	query := []float32{1, 2, 3}
	values := []float32{3, 2, 1}
	blob := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(blob[index*4:], math.Float32bits(value))
	}
	want, err := vectorutil.Cosine(query, values)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cosineBlob(query, blob, len(values))
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-want) > 1e-15 {
		t.Fatalf("cosineBlob = %.17g, vector.Cosine = %.17g", got, want)
	}
}

func TestCosineBlobRejectsInvalidAndNonFiniteVectors(t *testing.T) {
	if _, err := cosineBlob([]float32{1}, []byte{0, 0, 0}, 1); err == nil {
		t.Fatal("cosineBlob accepted a truncated blob")
	}
	nonFinite := make([]byte, 4)
	binary.LittleEndian.PutUint32(nonFinite, math.Float32bits(float32(math.Inf(1))))
	if _, err := cosineBlob([]float32{1}, nonFinite, 1); err == nil {
		t.Fatal("cosineBlob accepted a non-finite vector")
	}
}

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

func TestBuildEntriesMatchesEagerManifestAndFailsClosed(t *testing.T) {
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
	eagerPath := filepath.Join(t.TempDir(), "eager.sqlite")
	eager, err := Build(t.Context(), Config{Path: eagerPath, Model: "model"}, representations, chunks, vectors, fakeEmbedder{vector: []float32{1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	if err := eager.Close(); err != nil {
		t.Fatal(err)
	}
	want, err := Inspect(eagerPath)
	if err != nil {
		t.Fatal(err)
	}

	entries := []Entry{
		{RepresentationID: "rep-a", ChunkID: "chunk-a", DocumentID: "doc-a", Values: []float32{1, 0}, ContentDigest: "a"},
		{RepresentationID: "rep-b", ChunkID: "chunk-b", DocumentID: "doc-b", Values: []float32{0, 1}, ContentDigest: "b"},
	}
	streamedPath := filepath.Join(t.TempDir(), "streamed.sqlite")
	streamed, err := BuildEntries(t.Context(), Config{Path: streamedPath, Model: "model"}, len(entries), entryProducer(entries), fakeEmbedder{vector: []float32{1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	if err := streamed.Close(); err != nil {
		t.Fatal(err)
	}
	got, err := Inspect(streamedPath)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("streamed manifest = %#v, want %#v", got, want)
	}

	_, err = BuildEntries(t.Context(), Config{Path: filepath.Join(t.TempDir(), "unordered.sqlite"), Model: "model"}, len(entries), entryProducer([]Entry{entries[1], entries[0]}), fakeEmbedder{vector: []float32{1, 0}})
	if err == nil || !strings.Contains(err.Error(), "strictly increasing") {
		t.Fatalf("unordered error = %v", err)
	}
	_, err = BuildEntries(t.Context(), Config{Path: filepath.Join(t.TempDir(), "short.sqlite"), Model: "model"}, len(entries)+1, entryProducer(entries), fakeEmbedder{vector: []float32{1, 0}})
	if err == nil || !strings.Contains(err.Error(), "expected") {
		t.Fatalf("short producer error = %v", err)
	}
}

func entryProducer(entries []Entry) func(func(Entry) error) error {
	return func(yield func(Entry) error) error {
		for _, entry := range entries {
			if err := yield(entry); err != nil {
				return err
			}
		}
		return nil
	}
}

func TestSQLiteURIEncodesPathBeforeParameters(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vectors?#.sqlite")
	uri := sqliteURI(path, url.Values{"mode": {"ro"}})
	if strings.Contains(uri, "?#.sqlite?") || !strings.Contains(uri, "%3F%23") {
		t.Fatalf("SQLite URI %q does not encode path separators", uri)
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
