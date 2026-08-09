package hnswcandidate

import (
	"context"
	"math"
	"reflect"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	vectorutil "github.com/go-go-golems/ragkit/vector"
)

type fakeEmbedder struct{ vector []float32 }

func (f fakeEmbedder) Embed(context.Context, rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	return rag.EmbeddingResult{Vectors: [][]float32{f.vector}}, nil
}
func fixtureEntries() []Entry {
	return []Entry{{"rep-b", "chunk-b", "doc-b", []float32{0, 1}}, {"rep-a", "chunk-a", "doc-a", []float32{1, 0}}, {"rep-c", "chunk-c", "doc-c", []float32{-1, 0}}, {"rep-d", "chunk-d", "doc-d", []float32{0, -1}}}
}
func fixtureConfig() Config {
	return Config{Model: "model", M: 2, Ml: .25, EfConstruction: 20, EfSearch: 20, Seed: 1}
}

func TestBuildSearchHasDeterministicBoundedRecall(t *testing.T) {
	t.Parallel()
	entries := fixtureEntries()
	first, err := Build(fixtureConfig(), entries, fakeEmbedder{[]float32{1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	second, err := Build(fixtureConfig(), entries, fakeEmbedder{[]float32{1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	firstHits, err := first.SearchVector([]float32{1, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	secondHits, err := second.SearchVector([]float32{1, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(firstHits) > 2 {
		t.Fatalf("candidate bound = %d, want <= 2", len(firstHits))
	}
	if !reflect.DeepEqual(firstHits, secondHits) {
		t.Fatalf("seeded results differ: first=%#v second=%#v", firstHits, secondHits)
	}
	if len(firstHits) != 2 || firstHits[0].RepresentationID != "rep-a" || firstHits[0].Rank != 1 {
		t.Fatalf("unexpected hits: %#v", firstHits)
	}
	exact := exactIDs(t, entries, []float32{1, 0}, 2)
	if recall(exact, firstHits) != 1 {
		t.Fatalf("recall@2 = %v, want 1; exact=%v candidate=%v", recall(exact, firstHits), exact, firstHits)
	}
}
func TestBuildRejectsInconsistentDimensions(t *testing.T) {
	t.Parallel()
	_, err := Build(fixtureConfig(), []Entry{{"a", "a", "a", []float32{1, 0}}, {"b", "b", "b", []float32{1}}}, fakeEmbedder{})
	if err == nil {
		t.Fatal("Build() error = nil, want inconsistent dimensions")
	}
}

func TestBuildRejectsNonFiniteMl(t *testing.T) {
	t.Parallel()
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		config := fixtureConfig()
		config.Ml = value
		if _, err := Build(config, fixtureEntries(), fakeEmbedder{}); err == nil {
			t.Fatalf("Build(Ml=%v) error = nil", value)
		}
	}
}

func TestBuildOwnsEntryVectors(t *testing.T) {
	t.Parallel()
	entries := fixtureEntries()
	index, err := Build(fixtureConfig(), entries, fakeEmbedder{})
	if err != nil {
		t.Fatal(err)
	}
	entries[1].Values[0] = -1
	hits, err := index.SearchVector([]float32{1, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].RepresentationID != "rep-a" {
		t.Fatalf("hits after caller mutation = %#v, want rep-a", hits)
	}
}

func TestSearchVectorRejectsInvalidQueriesBeforeGraphSearch(t *testing.T) {
	t.Parallel()
	index, err := Build(fixtureConfig(), fixtureEntries(), fakeEmbedder{})
	if err != nil {
		t.Fatal(err)
	}
	for _, query := range [][]float32{{1}, {float32(math.NaN()), 0}, {float32(math.Inf(1)), 0}} {
		if _, err := index.SearchVector(query, 2); err == nil {
			t.Fatalf("SearchVector(%v) error = nil", query)
		}
	}
}
func exactIDs(t *testing.T, entries []Entry, query []float32, limit int) []string {
	t.Helper()
	hits := make([]rag.Hit, 0, len(entries))
	for _, entry := range entries {
		score, err := vectorutil.Cosine(query, entry.Values)
		if err != nil {
			t.Fatal(err)
		}
		hits = append(hits, rag.Hit{RepresentationID: entry.RepresentationID, ChunkID: entry.ChunkID, DocumentID: entry.DocumentID, Score: score})
	}
	for i := range hits {
		for j := i + 1; j < len(hits); j++ {
			if rag.HitRanksBefore(hits[j], hits[i]) {
				hits[i], hits[j] = hits[j], hits[i]
			}
		}
	}
	ids := make([]string, 0, limit)
	for _, hit := range hits[:limit] {
		ids = append(ids, hit.RepresentationID)
	}
	return ids
}
func recall(exact []string, candidate []rag.Hit) float64 {
	seen := map[string]bool{}
	for _, hit := range candidate {
		seen[hit.RepresentationID] = true
	}
	n := 0
	for _, id := range exact {
		if seen[id] {
			n++
		}
	}
	return float64(n) / float64(len(exact))
}
