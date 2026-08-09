package rag

import (
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/stretchr/testify/require"
)

func TestEvidenceIdentitiesTrackOrderedSourceContentButIgnoreScores(t *testing.T) {
	evidence := []Evidence{
		{
			Chunk:          Chunk{ID: "c1", Text: "first", ContentDigest: digest.Text("first")},
			Rank:           1,
			RetrievalScore: 1.996638799554313,
		},
		{
			Chunk:          Chunk{ID: "c2", Text: "second", ContentDigest: digest.Text("second")},
			Rank:           2,
			RetrievalScore: 1.25,
		},
	}
	first, err := EvidenceIdentities(evidence)
	require.NoError(t, err)

	rerankerScore := 0.75
	evidence[0].RetrievalScore = 1.9966387995543127
	evidence[0].RerankerScore = &rerankerScore
	evidence[0].Rank = 99
	scoreChanged, err := EvidenceIdentities(evidence)
	require.NoError(t, err)
	require.Equal(t, first, scoreChanged)

	evidence[0], evidence[1] = evidence[1], evidence[0]
	reordered, err := EvidenceIdentities(evidence)
	require.NoError(t, err)
	require.NotEqual(t, first, reordered)

	evidence[0].Chunk.Text = "changed"
	contentChanged, err := EvidenceIdentities(evidence)
	require.ErrorContains(t, err, "content digest mismatch")
	require.Nil(t, contentChanged)
}

func TestEvidenceIdentitiesDeriveMissingContentDigest(t *testing.T) {
	identities, err := EvidenceIdentities([]Evidence{{
		Chunk: Chunk{ID: "c1", Text: "source text"},
	}})
	require.NoError(t, err)
	require.Len(t, identities, 1)
	require.Equal(t, digest.Text("source text"), identities[0].ContentDigest)
}

func TestEvidenceIdentitiesRejectStaleSuppliedDigest(t *testing.T) {
	identities, err := EvidenceIdentities([]Evidence{{
		Chunk: Chunk{ID: "c1", Text: "current source text", ContentDigest: digest.Text("old source text")},
	}})
	require.ErrorContains(t, err, "content digest mismatch")
	require.Nil(t, identities)
}
