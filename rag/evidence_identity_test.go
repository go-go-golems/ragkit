package rag

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEvidenceIdentitiesTrackOrderedSourceContentButIgnoreScores(t *testing.T) {
	evidence := []Evidence{
		{
			Chunk:          Chunk{ID: "c1", Text: "first", ContentDigest: "digest-1"},
			Rank:           1,
			RetrievalScore: 1.996638799554313,
		},
		{
			Chunk:          Chunk{ID: "c2", Text: "second", ContentDigest: "digest-2"},
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

	evidence[0].Chunk.ContentDigest = "changed"
	contentChanged, err := EvidenceIdentities(evidence)
	require.NoError(t, err)
	require.NotEqual(t, reordered, contentChanged)
}

func TestEvidenceIdentitiesDeriveMissingContentDigest(t *testing.T) {
	identities, err := EvidenceIdentities([]Evidence{{
		Chunk: Chunk{ID: "c1", Text: "source text"},
	}})
	require.NoError(t, err)
	require.Len(t, identities, 1)
	require.NotEmpty(t, identities[0].ContentDigest)
}
