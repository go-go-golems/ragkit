package bleve

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
)

func TestBuildRecordsMatchesEagerManifestAndFailsClosed(t *testing.T) {
	documents, chunks, representations := fixture()
	eager, want, err := Build(t.Context(), Config{Path: filepath.Join(t.TempDir(), "eager")}, documents, chunks, representations)
	if err != nil {
		t.Fatal(err)
	}
	requireClose(t, eager)
	records := []Record{
		{RepresentationID: "rep-oak", ChunkID: "chunk-oak", DocumentID: "doc-oak", Kind: "raw", Title: "Oak planting", Body: chunks[0].Text},
		{RepresentationID: "rep-maple", ChunkID: "chunk-maple", DocumentID: "doc-maple", Kind: "raw", Title: "Maple color", Body: chunks[1].Text},
	}
	sort.Slice(records, func(i, j int) bool { return records[i].RepresentationID < records[j].RepresentationID })
	streamed, got, err := BuildRecords(t.Context(), Config{Path: filepath.Join(t.TempDir(), "streamed"), BatchSize: 1}, len(records), recordProducer(records))
	if err != nil {
		t.Fatal(err)
	}
	requireClose(t, streamed)
	if got != want {
		t.Fatalf("streamed manifest = %#v, want %#v", got, want)
	}

	_, _, err = BuildRecords(t.Context(), Config{Path: filepath.Join(t.TempDir(), "unordered")}, len(records), recordProducer([]Record{records[1], records[0]}))
	if err == nil || !strings.Contains(err.Error(), "strictly increasing") {
		t.Fatalf("unordered error = %v", err)
	}
	_, _, err = BuildRecords(t.Context(), Config{Path: filepath.Join(t.TempDir(), "short")}, len(records)+1, recordProducer(records))
	if err == nil || !strings.Contains(err.Error(), "expected") {
		t.Fatalf("short producer error = %v", err)
	}
}

func recordProducer(records []Record) func(func(Record) error) error {
	return func(yield func(Record) error) error {
		for _, current := range records {
			if err := yield(current); err != nil {
				return err
			}
		}
		return nil
	}
}

func requireClose(t *testing.T, index *Index) {
	t.Helper()
	if err := index.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestBuildSearchReopen(t *testing.T) {
	t.Parallel()
	documents, chunks, representations := fixture()
	path := filepath.Join(t.TempDir(), "bm25.bleve")
	index, manifest, err := Build(context.Background(), Config{Path: path}, documents, chunks, representations)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.RepresentationCount != 2 {
		t.Fatalf("representation count = %d", manifest.RepresentationCount)
	}
	if manifest.Version != ManifestVersion {
		t.Fatalf("manifest version = %d, want %d", manifest.Version, ManifestVersion)
	}
	hits, err := index.Search(context.Background(), rag.Query{Text: "oak soil"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].ChunkID != "chunk-oak" || hits[0].DocumentID != "doc-oak" {
		t.Fatalf("unexpected hits: %#v", hits)
	}
	if err := index.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := Open(path, "")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = reopened.Close() }()
	reopenedHits, err := reopened.Search(context.Background(), rag.Query{Text: "oak soil"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(reopenedHits) != len(hits) || reopenedHits[0].RepresentationID != hits[0].RepresentationID {
		t.Fatalf("reopen mismatch: %#v != %#v", reopenedHits, hits)
	}
}

func TestBuildRejectsInvalidBoostsBeforeCreatingOutput(t *testing.T) {
	documents, chunks, representations := fixture()
	for name, boost := range map[string]float64{
		"nan": math.NaN(), "positive-infinity": math.Inf(1), "negative-infinity": math.Inf(-1), "negative": -1,
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "invalid.bleve")
			_, _, err := Build(t.Context(), Config{Path: path, TitleBoost: boost}, documents, chunks, representations)
			if err == nil {
				t.Fatal("Build() error = nil")
			}
			if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
				t.Fatalf("output exists after validation failure: %v", statErr)
			}
		})
	}
}

func TestBuildRejectsDuplicateRepresentationIDsBeforeCreatingOutput(t *testing.T) {
	t.Parallel()
	documents, chunks, representations := fixture()
	representations[1].ID = representations[0].ID
	path := filepath.Join(t.TempDir(), "duplicate.bleve")
	_, _, err := Build(t.Context(), Config{Path: path}, documents, chunks, representations)
	if err == nil {
		t.Fatal("Build() error = nil")
	}
	if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
		t.Fatalf("output exists after validation failure: %v", statErr)
	}
}

func TestSearchOrdersTiesBeforeApplyingLimit(t *testing.T) {
	t.Parallel()
	const (
		count = 30
		limit = 20
	)
	documents := make([]rag.Document, 0, count)
	chunks := make([]rag.Chunk, 0, count)
	representations := make([]rag.Representation, 0, count)
	// Insert in descending identity order. All records have identical title and
	// body statistics, producing an equal-score group larger than the cutoff.
	// Chunk IDs deliberately sort opposite to document IDs so the test proves
	// that the complete HitRanksBefore identity controls the provider cutoff.
	for index := count - 1; index >= 0; index-- {
		representationID := fmt.Sprintf("rep-%02d", index)
		chunkID := fmt.Sprintf("chunk-%02d", count-1-index)
		documentID := fmt.Sprintf("doc-%02d", index)
		documents = append(documents, rag.Document{ID: documentID, Title: "Tied title"})
		chunks = append(chunks, rag.Chunk{ID: chunkID, DocumentID: documentID, Text: "identical token"})
		representations = append(representations, rag.Representation{
			ID: representationID, ChunkID: chunkID, Kind: "raw",
			Text: "identical token", ContentDigest: representationID,
		})
	}
	path := filepath.Join(t.TempDir(), "ties.bleve")
	index, _, err := Build(context.Background(), Config{Path: path}, documents, chunks, representations)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = index.Close() }()

	for repetition := 0; repetition < 3; repetition++ {
		hits, err := index.Search(context.Background(), rag.Query{Text: "identical token"}, limit)
		if err != nil {
			t.Fatal(err)
		}
		if len(hits) != limit {
			t.Fatalf("hits = %d, want %d", len(hits), limit)
		}
		for position, hit := range hits {
			wantChunk := fmt.Sprintf("chunk-%02d", count-1-position)
			wantRepresentation := fmt.Sprintf("rep-%02d", position)
			if hit.ChunkID != wantChunk || hit.RepresentationID != wantRepresentation {
				t.Fatalf("repetition %d rank %d = (%q, %q), want (%q, %q)", repetition, position+1, hit.ChunkID, hit.RepresentationID, wantChunk, wantRepresentation)
			}
			if hit.Rank != position+1 {
				t.Fatalf("repetition %d rank field = %d, want %d", repetition, hit.Rank, position+1)
			}
		}
	}
}

func fixture() ([]rag.Document, []rag.Chunk, []rag.Representation) {
	documents := []rag.Document{
		{ID: "doc-oak", Title: "Oak planting"},
		{ID: "doc-maple", Title: "Maple color"},
	}
	chunks := []rag.Chunk{
		{ID: "chunk-oak", DocumentID: "doc-oak", Text: "oak trees like soil"},
		{ID: "chunk-maple", DocumentID: "doc-maple", Text: "maple leaves are red"},
	}
	representations := []rag.Representation{
		{ID: "rep-oak", ChunkID: "chunk-oak", Kind: "raw", Text: chunks[0].Text, ContentDigest: "oak"},
		{ID: "rep-maple", ChunkID: "chunk-maple", Kind: "raw", Text: chunks[1].Text, ContentDigest: "maple"},
	}
	return documents, chunks, representations
}
