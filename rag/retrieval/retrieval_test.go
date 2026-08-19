package retrieval

import (
	"fmt"
	"math"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/go-go-golems/ragkit/rag/content"
	"github.com/stretchr/testify/require"
)

func TestHydrateFromStoreLoadsOnlyBoundedCandidates(t *testing.T) {
	store, err := content.NewMemory(nil, []rag.Chunk{
		{ID: "a", DocumentID: "doc-a", Text: "alpha"},
		{ID: "b", DocumentID: "doc-b", Text: "bravo"},
		{ID: "c", DocumentID: "doc-c", Text: "charlie"},
	})
	require.NoError(t, err)
	evidence, err := HydrateFromStore(t.Context(), []rag.FusedHit{
		{ChunkID: "b", Score: 0.9}, {ChunkID: "a", Score: 0.8}, {ChunkID: "c", Score: 0.7},
	}, store, 2)
	require.NoError(t, err)
	require.Equal(t, []string{"b", "a"}, []string{evidence[0].Chunk.ID, evidence[1].Chunk.ID})
	require.Equal(t, []int{1, 2}, []int{evidence[0].Rank, evidence[1].Rank})

	_, err = HydrateFromStore(t.Context(), []rag.FusedHit{{ChunkID: "missing"}}, store, 1)
	require.ErrorContains(t, err, "missing")
	_, err = HydrateFromStore(t.Context(), []rag.FusedHit{{ChunkID: "a"}, {ChunkID: "a"}}, store, 2)
	require.ErrorContains(t, err, "duplicate fused chunk")
}

func TestHydrateFromStoreSplitsLargeCandidateSets(t *testing.T) {
	chunks := make([]rag.Chunk, content.DefaultMaxBatch+1)
	hits := make([]rag.FusedHit, len(chunks))
	for index := range chunks {
		id := fmt.Sprintf("chunk-%03d", index)
		chunks[index] = rag.Chunk{ID: id, DocumentID: "doc", Text: id}
		hits[index] = rag.FusedHit{ChunkID: id, Rank: index + 1, Score: float64(len(chunks) - index)}
	}
	store, err := content.NewMemory(nil, chunks)
	require.NoError(t, err)
	evidence, err := HydrateFromStore(t.Context(), hits, store, len(hits))
	require.NoError(t, err)
	require.Len(t, evidence, len(hits))
	require.Equal(t, hits[len(hits)-1].ChunkID, evidence[len(evidence)-1].Chunk.ID)
}

func TestCollapseAndRRF(t *testing.T) {
	t.Parallel()
	lexical := []rag.Hit{{ChunkID: "a", DocumentID: "doc-a", Rank: 1}, {ChunkID: "a", DocumentID: "doc-a", Rank: 2}}
	collapsed, err := Collapse(lexical, TargetChunk)
	if err != nil {
		t.Fatalf("Collapse() error = %v", err)
	}
	if len(collapsed) != 1 {
		t.Fatalf("collapsed = %d, want 1", len(collapsed))
	}
	fused, err := WeightedRRF(map[string][]rag.Hit{
		"lexical": collapsed,
		"vector":  {{ChunkID: "b", DocumentID: "doc-b", Rank: 1}},
	}, RRFConfig{RankConstant: 60})
	if err != nil {
		t.Fatalf("WeightedRRF() error = %v", err)
	}
	if len(fused) != 2 {
		t.Fatalf("fused = %d, want 2", len(fused))
	}
}

func TestWeightedRRFRejectsNonFiniteConfiguration(t *testing.T) {
	t.Parallel()
	channels := map[string][]rag.Hit{"lexical": {{ChunkID: "chunk", Rank: 1}}}
	for _, rankConstant := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		_, err := WeightedRRF(channels, RRFConfig{RankConstant: rankConstant})
		if err == nil {
			t.Fatalf("WeightedRRF rank constant %v error = nil", rankConstant)
		}
	}
	for _, weight := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		_, err := WeightedRRF(channels, RRFConfig{RankConstant: 60, Weights: map[string]float64{"lexical": weight}})
		if err == nil {
			t.Fatalf("WeightedRRF weight %v error = nil", weight)
		}
	}
}

func TestWeightedRRFRejectsNegativeWeight(t *testing.T) {
	_, err := WeightedRRF(map[string][]rag.Hit{"lexical": {{ChunkID: "c", Rank: 1}}}, RRFConfig{
		RankConstant: 60, Weights: map[string]float64{"lexical": -1},
	})
	require.ErrorContains(t, err, "non-negative")
}

func TestWeightedRRFRejectsInvalidRanks(t *testing.T) {
	for _, rank := range []int{0, -1} {
		_, err := WeightedRRF(map[string][]rag.Hit{
			"lexical": {{ChunkID: "c", Rank: rank}},
		}, RRFConfig{RankConstant: 60})
		require.ErrorContains(t, err, "invalid rank")
	}
}

func TestWeightedRRFRejectsConflictingChunkDocuments(t *testing.T) {
	_, err := WeightedRRF(map[string][]rag.Hit{
		"lexical": {{ChunkID: "chunk", DocumentID: "doc-a", Rank: 1}},
		"vector":  {{ChunkID: "chunk", DocumentID: "doc-b", Rank: 1}},
	}, RRFConfig{RankConstant: 60})
	require.ErrorContains(t, err, "conflicting document identities")
}

func TestWeightedRRFUsesChunkIDAsFinalTieBreaker(t *testing.T) {
	t.Parallel()
	fused, err := WeightedRRF(map[string][]rag.Hit{
		"lexical": {
			{ChunkID: "chunk-z", DocumentID: "doc-z", Rank: 1},
			{ChunkID: "chunk-a", DocumentID: "doc-a", Rank: 1},
		},
	}, RRFConfig{RankConstant: 60})
	if err != nil {
		t.Fatal(err)
	}
	if len(fused) != 2 {
		t.Fatalf("fused = %d, want 2", len(fused))
	}
	if fused[0].ChunkID != "chunk-a" || fused[1].ChunkID != "chunk-z" {
		t.Fatalf("ordered IDs = [%s %s], want [chunk-a chunk-z]", fused[0].ChunkID, fused[1].ChunkID)
	}
}

func TestCollapseUsesChunkIDBeforeRepresentationIDForTiedRanks(t *testing.T) {
	t.Parallel()
	collapsed, err := Collapse([]rag.Hit{
		{ChunkID: "chunk-z", RepresentationID: "rep-a", Rank: 1},
		{ChunkID: "chunk-a", RepresentationID: "rep-z", Rank: 1},
	}, TargetChunk)
	if err != nil {
		t.Fatal(err)
	}
	if collapsed[0].ChunkID != "chunk-a" || collapsed[1].ChunkID != "chunk-z" {
		t.Fatalf("ordered IDs = [%s %s], want [chunk-a chunk-z]", collapsed[0].ChunkID, collapsed[1].ChunkID)
	}
}

func TestCollapseRejectsNonFiniteHitScores(t *testing.T) {
	for _, score := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		_, err := Collapse([]rag.Hit{{ChunkID: "chunk", Rank: 1, Score: score}}, TargetChunk)
		require.ErrorContains(t, err, "non-finite score")
	}
}
