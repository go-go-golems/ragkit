package retrieval

import (
	"math"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

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
