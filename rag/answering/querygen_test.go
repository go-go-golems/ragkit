package answering

import (
	"context"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

// recordingSearcher keeps every query it saw, so a test can assert which
// variants actually reached each channel.
type recordingSearcher struct {
	hits    []rag.Hit
	queries []rag.Query
}

func (s *recordingSearcher) Search(_ context.Context, query rag.Query, _ int) ([]rag.Hit, error) {
	s.queries = append(s.queries, query)
	return append([]rag.Hit(nil), s.hits...), nil
}

func querygenFixture(generated string, generr error) (*Service, *recordingSearcher, *recordingSearcher) {
	lexical := &recordingSearcher{hits: []rag.Hit{
		{RepresentationID: "rep-a", ChunkID: "chunk-a", DocumentID: "doc-a", Channel: "bm25", Rank: 1, Score: 1},
	}}
	vector := &recordingSearcher{hits: []rag.Hit{
		{RepresentationID: "rep-b", ChunkID: "chunk-b", DocumentID: "doc-b", Channel: "vector", Rank: 1, Score: 0.9},
	}}
	return &Service{
		Lexical: lexical, Vector: vector,
		Generator: &fixedGenerator{
			result: rag.GenerationResult{Text: generated}, err: generr,
		},
		Chunks: []rag.Chunk{
			{ID: "chunk-a", DocumentID: "doc-a", Text: "a"},
			{ID: "chunk-b", DocumentID: "doc-b", Text: "b"},
		},
		GenerationModel: "generator",
	}, lexical, vector
}

func TestMultiQueryFansOutAndRecordsVariants(t *testing.T) {
	service, lexical, vector := querygenFixture(
		`{"variants": ["prune timing for leyland", "leyland cypress trim season"]}`, nil,
	)
	request := requestFixture(StrategyMultiQuery)
	request.Config.QueryVariants = 2

	result, err := service.Retrieve(context.Background(), request)
	require.NoError(t, err)
	require.Equal(t, []string{
		"prune timing for leyland", "leyland cypress trim season",
	}, result.Variants)
	require.Empty(t, result.VariantError)
	// Original plus two variants, per channel.
	require.Len(t, lexical.queries, 3)
	require.Len(t, vector.queries, 3)
	require.Equal(t, "question", lexical.queries[0].Text)
	require.Equal(t, "query-v1", lexical.queries[1].ID,
		"variants keep the source query id with a suffix")
	require.Contains(t, result.Channels, "bm25:q0")
	require.Contains(t, result.Channels, "vector:q2")
	require.NotEmpty(t, result.Fused)
}

func TestMultiQueryFallsBackWhenTheContractFails(t *testing.T) {
	service, lexical, _ := querygenFixture("this is not json", nil)
	result, err := service.Retrieve(
		context.Background(), requestFixture(StrategyMultiQuery),
	)
	require.NoError(t, err, "reformulation failure must not fail the turn")
	require.Empty(t, result.Variants)
	require.Contains(t, result.VariantError, "decode variants")
	require.Len(t, lexical.queries, 1, "only the plain question retrieves")
	require.NotEmpty(t, result.Fused)
}

func TestHyDEUsesTheHypotheticalOnTheVectorChannelOnly(t *testing.T) {
	service, lexical, vector := querygenFixture(
		"Prune Leyland Cypress in early spring before growth begins.", nil,
	)
	result, err := service.Retrieve(
		context.Background(), requestFixture(StrategyHyDE),
	)
	require.NoError(t, err)
	require.Len(t, result.Variants, 1)
	require.Equal(t, "question", lexical.queries[0].Text,
		"the lexical channel keeps the question")
	require.Contains(t, vector.queries[0].Text, "early spring",
		"the vector channel searches the hypothetical answer")
	require.Empty(t, result.VariantError)
}

func TestHyDEFallsBackToTheQuestionOnEmptyGeneration(t *testing.T) {
	service, _, vector := querygenFixture("", nil)
	result, err := service.Retrieve(
		context.Background(), requestFixture(StrategyHyDE),
	)
	require.NoError(t, err)
	require.Contains(t, result.VariantError, "empty hypothetical")
	require.Equal(t, "question", vector.queries[0].Text)
}
