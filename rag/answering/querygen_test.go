package answering

import (
	"context"
	"errors"
	"strings"
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
		Content: memoryContent([]rag.Chunk{
			{ID: "chunk-a", DocumentID: "doc-a", Text: "a"},
			{ID: "chunk-b", DocumentID: "doc-b", Text: "b"},
		}),
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

func TestMultiQueryFallsBackUnlessAllVariantsAreDistinctAndUsable(t *testing.T) {
	service, lexical, _ := querygenFixture(
		`{"variants":["question","distinct","distinct",""]}`, nil,
	)
	request := requestFixture(StrategyMultiQuery)
	request.Config.QueryVariants = 2
	result, err := service.Retrieve(t.Context(), request)
	require.NoError(t, err)
	require.Empty(t, result.Variants)
	require.Contains(t, result.VariantError, "1 distinct usable variants; expected 2")
	require.Len(t, lexical.queries, 1, "partial reformulations must not change retrieval")
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

func TestQueryGenerationObserverFailureStopsRetrieval(t *testing.T) {
	for _, strategy := range []Strategy{StrategyMultiQuery, StrategyHyDE} {
		t.Run(string(strategy), func(t *testing.T) {
			generated := `{"variants":["variant-1","variant-2","variant-3"]}`
			if strategy == StrategyHyDE {
				generated = "hypothetical answer"
			}
			service, lexical, vector := querygenFixture(generated, nil)
			service.Observer = func(_ context.Context, observation Observation) error {
				if observation.Stage == StageQueryGen && observation.Status == ObservationCompleted {
					return errors.New("observer unavailable")
				}
				return nil
			}

			_, err := service.Retrieve(context.Background(), requestFixture(strategy))
			require.ErrorContains(t, err, "observer unavailable")
			require.Empty(t, lexical.queries)
			require.Empty(t, vector.queries)
		})
	}
}

func TestQueryGenerationFallbackObserverFailureStopsRetrieval(t *testing.T) {
	for _, strategy := range []Strategy{StrategyMultiQuery, StrategyHyDE} {
		t.Run(string(strategy), func(t *testing.T) {
			generated := "not json"
			if strategy == StrategyHyDE {
				generated = ""
			}
			service, lexical, vector := querygenFixture(generated, nil)
			service.Observer = func(_ context.Context, observation Observation) error {
				if observation.Stage == StageQueryGen && observation.Status == ObservationFailed {
					return errors.New("failure observation unavailable")
				}
				return nil
			}
			_, err := service.Retrieve(t.Context(), requestFixture(strategy))
			require.ErrorContains(t, err, "failure observation unavailable")
			require.Empty(t, lexical.queries)
			require.Empty(t, vector.queries)
		})
	}
}

func TestQueryGenerationUsageSurvivesSuccessAndFallback(t *testing.T) {
	for name, generated := range map[string]string{
		"success":  `{"variants":["one","two","three"]}`,
		"fallback": "not json",
	} {
		t.Run(name, func(t *testing.T) {
			service, _, _ := querygenFixture(generated, nil)
			tokens := int64(17)
			service.Generator.(*fixedGenerator).result.Usage = rag.Usage{InputTokens: &tokens}
			result, err := service.Retrieve(t.Context(), requestFixture(StrategyMultiQuery))
			require.NoError(t, err)
			require.NotNil(t, result.QueryGenerationUsage.InputTokens)
			require.Equal(t, tokens, *result.QueryGenerationUsage.InputTokens)
		})
	}
}

func TestDefaultQueryTransformationPromptsAreDomainNeutral(t *testing.T) {
	service, _, _ := querygenFixture(`{"variants":["one","two","three"]}`, nil)
	_, err := service.Retrieve(t.Context(), requestFixture(StrategyMultiQuery))
	require.NoError(t, err)
	prompt := service.Generator.(*fixedGenerator).request.Prompt
	require.NotContains(t, strings.ToLower(prompt), "plant")
	require.Contains(t, prompt, "domain-specific entities")

	service, _, _ = querygenFixture("hypothetical", nil)
	_, err = service.Retrieve(t.Context(), requestFixture(StrategyHyDE))
	require.NoError(t, err)
	prompt = service.Generator.(*fixedGenerator).request.Prompt
	require.NotContains(t, strings.ToLower(prompt), "plant")
	require.Contains(t, prompt, "domain of the question")
}
