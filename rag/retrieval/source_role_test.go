package retrieval

import (
	"context"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func TestSourceRoleSearcherFiltersBeforeTopK(t *testing.T) {
	underlying := &recordingSearcher{hits: []rag.Hit{
		{RepresentationID: "product-a", DocumentID: "product", Rank: 1},
		{RepresentationID: "guide-a", DocumentID: "guide", Rank: 2},
		{RepresentationID: "faq-a", DocumentID: "faq", Rank: 3},
		{RepresentationID: "guide-b", DocumentID: "guide", Rank: 4},
	}}
	searcher, err := NewSourceRoleSearcher(underlying, []rag.Document{
		{ID: "product", Metadata: map[string]string{"source_role": "product"}},
		{ID: "guide", Metadata: map[string]string{"source_role": "ttc_guide"}},
		{ID: "faq", Metadata: map[string]string{"source_role": "faq"}},
	}, 4, "ttc_guide", "faq")
	require.NoError(t, err)

	hits, err := searcher.Search(context.Background(), rag.Query{Text: "watering"}, 2)
	require.NoError(t, err)
	require.Equal(t, 4, underlying.topK)
	require.Equal(t, []rag.Hit{
		{RepresentationID: "guide-a", DocumentID: "guide", Rank: 1},
		{RepresentationID: "faq-a", DocumentID: "faq", Rank: 2},
	}, hits)
}

func TestSourceRoleSearcherRejectsUnknownDocument(t *testing.T) {
	underlying := &recordingSearcher{hits: []rag.Hit{{RepresentationID: "unknown", DocumentID: "missing", Rank: 1}}}
	searcher, err := NewSourceRoleSearcher(underlying, []rag.Document{{ID: "guide", Metadata: map[string]string{"source_role": "ttc_guide"}}}, 1, "ttc_guide")
	require.NoError(t, err)

	_, err = searcher.Search(context.Background(), rag.Query{Text: "watering"}, 1)
	require.EqualError(t, err, `search result references unknown document "missing"`)
}

func TestSourceRoleSearcherValidatesConfiguration(t *testing.T) {
	underlying := &recordingSearcher{}
	documents := []rag.Document{{ID: "guide", Metadata: map[string]string{"source_role": "ttc_guide"}}}
	_, err := NewSourceRoleSearcher(nil, documents, 1, "ttc_guide")
	require.EqualError(t, err, "source-role searcher requires an underlying searcher")
	_, err = NewSourceRoleSearcher(underlying, documents, 0, "ttc_guide")
	require.EqualError(t, err, "source-role searcher requires a positive search depth")
	_, err = NewSourceRoleSearcher(underlying, documents, 1)
	require.EqualError(t, err, "source-role searcher requires at least one allowed role")
	_, err = NewSourceRoleSearcher(underlying, documents, 1, "product")
	require.ErrorContains(t, err, "has no documents")
}
