package retrieval

import (
	"context"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

type recordingSearcher struct {
	hits []rag.Hit
	topK int
}

func (s *recordingSearcher) Search(_ context.Context, _ rag.Query, topK int) ([]rag.Hit, error) {
	s.topK = topK
	if topK > len(s.hits) {
		topK = len(s.hits)
	}
	return append([]rag.Hit(nil), s.hits[:topK]...), nil
}

func TestRepresentationKindSearcherFiltersBeforeTopK(t *testing.T) {
	underlying := &recordingSearcher{hits: []rag.Hit{
		{RepresentationID: "raw-a", Rank: 1},
		{RepresentationID: "summary-a", Rank: 2},
		{RepresentationID: "raw-b", Rank: 3},
		{RepresentationID: "summary-b", Rank: 4},
	}}
	searcher, err := NewRepresentationKindSearcher(underlying, []rag.Representation{
		{ID: "raw-a", Kind: "raw"},
		{ID: "summary-a", Kind: "summary"},
		{ID: "raw-b", Kind: "raw"},
		{ID: "summary-b", Kind: "summary"},
	}, "summary")
	require.NoError(t, err)

	hits, err := searcher.Search(context.Background(), rag.Query{Text: "query"}, 2)
	require.NoError(t, err)
	require.Equal(t, 4, underlying.topK, "the complete index must be searched before filtering")
	require.Equal(t, []rag.Hit{
		{RepresentationID: "summary-a", Rank: 1},
		{RepresentationID: "summary-b", Rank: 2},
	}, hits)
}

func TestRepresentationKindSearcherRejectsUnknownHitIdentity(t *testing.T) {
	underlying := &recordingSearcher{hits: []rag.Hit{{RepresentationID: "unknown", Rank: 1}}}
	searcher, err := NewRepresentationKindSearcher(underlying, []rag.Representation{{ID: "raw-a", Kind: "raw"}}, "raw")
	require.NoError(t, err)

	_, err = searcher.Search(context.Background(), rag.Query{Text: "query"}, 1)
	require.EqualError(t, err, `search result references unknown representation "unknown"`)
}

func TestRepresentationKindSearcherValidatesConfiguration(t *testing.T) {
	underlying := &recordingSearcher{}
	_, err := NewRepresentationKindSearcher(nil, []rag.Representation{{ID: "raw-a", Kind: "raw"}}, "raw")
	require.EqualError(t, err, "representation-kind searcher requires an underlying searcher")
	_, err = NewRepresentationKindSearcher(underlying, nil, "raw")
	require.EqualError(t, err, "representation-kind searcher requires representations")
	_, err = NewRepresentationKindSearcher(underlying, []rag.Representation{{ID: "raw-a", Kind: "raw"}})
	require.EqualError(t, err, "representation-kind searcher requires at least one allowed kind")
	_, err = NewRepresentationKindSearcher(underlying, []rag.Representation{{ID: "raw-a", Kind: "raw"}}, "summary")
	require.ErrorContains(t, err, "has no representations")
}
