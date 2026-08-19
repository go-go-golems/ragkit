package answering

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/go-go-golems/ragkit/rag/content"
	"github.com/stretchr/testify/require"
)

type fixedSearcher struct {
	hits  []rag.Hit
	query rag.Query
	calls int
}

func (s *fixedSearcher) Search(_ context.Context, query rag.Query, _ int) ([]rag.Hit, error) {
	s.calls++
	s.query = query
	return append([]rag.Hit(nil), s.hits...), nil
}

func TestRRFRejectsNonFiniteConstantsBeforeSearch(t *testing.T) {
	for name, constant := range map[string]float64{
		"nan": math.NaN(), "positive infinity": math.Inf(1), "negative infinity": math.Inf(-1),
	} {
		t.Run(name, func(t *testing.T) {
			service, lexical, vector := serviceFixture()
			request := requestFixture(StrategyRRF)
			request.Config.RRFConstant = constant
			_, err := service.Retrieve(t.Context(), request)
			require.ErrorContains(t, err, "RRF constant must be finite")
			require.Zero(t, lexical.calls)
			require.Zero(t, vector.calls)
		})
	}
}

func TestRRFRejectsInvalidWeightsBeforeSearch(t *testing.T) {
	for name, weight := range map[string]float64{
		"nan": math.NaN(), "positive infinity": math.Inf(1), "negative infinity": math.Inf(-1), "negative": -1, "zero": 0,
	} {
		t.Run(name, func(t *testing.T) {
			service, lexical, vector := serviceFixture()
			service.Generator = &fixedGenerator{}
			request := requestFixture(StrategyMultiQuery)
			request.Config.RRFWeights = map[string]float64{"bm25": weight}
			_, err := service.Retrieve(t.Context(), request)
			require.ErrorContains(t, err, "RRF weight")
			require.Zero(t, lexical.calls)
			require.Zero(t, vector.calls)
		})
	}
}

func TestRRFRejectsUnknownWeightChannelsBeforeSearch(t *testing.T) {
	service, lexical, vector := serviceFixture()
	request := requestFixture(StrategyRRF)
	request.Config.RRFWeights = map[string]float64{"vectro": 2}
	_, err := service.Retrieve(t.Context(), request)
	require.ErrorContains(t, err, "unknown RRF weight channel")
	require.Zero(t, lexical.calls)
	require.Zero(t, vector.calls)
}

type reversingReranker struct{ candidates int }

func (r *reversingReranker) Rerank(_ context.Context, request rag.RerankRequest) (rag.RerankResult, error) {
	r.candidates = len(request.Candidates)
	result := append([]rag.Evidence(nil), request.Candidates...)
	for left, right := 0, len(result)-1; left < right; left, right = left+1, right-1 {
		result[left], result[right] = result[right], result[left]
	}
	if len(result) > request.Results {
		result = result[:request.Results]
	}
	for i := range result {
		result[i].Rank = i + 1
	}
	return rag.RerankResult{Evidence: result}, nil
}

type fixedReranker struct{ evidence []rag.Evidence }

func (r fixedReranker) Rerank(context.Context, rag.RerankRequest) (rag.RerankResult, error) {
	return rag.RerankResult{Evidence: r.evidence}, nil
}

type fixedGenerator struct {
	request rag.GenerationRequest
	result  rag.GenerationResult
	err     error
}

func (g *fixedGenerator) Generate(_ context.Context, request rag.GenerationRequest) (rag.GenerationResult, error) {
	g.request = request
	return g.result, g.err
}

func memoryContent(chunks []rag.Chunk) content.Store {
	store, err := content.NewMemory(nil, chunks)
	if err != nil {
		panic(err)
	}
	return store
}

func serviceFixture() (*Service, *fixedSearcher, *fixedSearcher) {
	lexical := &fixedSearcher{hits: []rag.Hit{
		{RepresentationID: "rep-a-summary", ChunkID: "chunk-a", DocumentID: "doc-a", Channel: "bm25", Rank: 2, Score: 0.8},
		{RepresentationID: "rep-a-raw", ChunkID: "chunk-a", DocumentID: "doc-a", Channel: "bm25", Rank: 1, Score: 1},
		{RepresentationID: "rep-b", ChunkID: "chunk-b", DocumentID: "doc-b", Channel: "bm25", Rank: 3, Score: 0.5},
	}}
	vector := &fixedSearcher{hits: []rag.Hit{
		{RepresentationID: "rep-b", ChunkID: "chunk-b", DocumentID: "doc-b", Channel: "vector", Rank: 1, Score: 0.9},
		{RepresentationID: "rep-c", ChunkID: "chunk-c", DocumentID: "doc-c", Channel: "vector", Rank: 2, Score: 0.7},
	}}
	return &Service{
		Lexical: lexical,
		Vector:  vector,
		Content: memoryContent([]rag.Chunk{
			{ID: "chunk-a", DocumentID: "doc-a", Text: "a"},
			{ID: "chunk-b", DocumentID: "doc-b", Text: "b"},
			{ID: "chunk-c", DocumentID: "doc-c", Text: "c"},
		}),
		GenerationModel: "generator",
		Prompt:          "answer from evidence",
		OutputSchema:    `{"type":"object"}`,
	}, lexical, vector
}

func requestFixture(strategy Strategy) Request {
	return Request{
		TurnID: "turn-1",
		Query:  rag.Query{ID: "query", Text: "question"},
		Config: RetrievalConfig{
			Strategy:            strategy,
			RetrieveK:           10,
			EvidenceK:           2,
			RerankCandidates:    3,
			RRFConstant:         60,
			MaximumContextRunes: 100,
		},
	}
}

func TestGenerateVariantsUsesCustomPromptVerbatim(t *testing.T) {
	t.Parallel()
	generator := &fixedGenerator{result: rag.GenerationResult{Text: `{"variants":["rewrite"]}`}}
	service, _, _ := serviceFixture()
	service.Generator = generator
	service.MultiQueryPrompt = "keep 100% of literal %s text"
	variants, _, _, err := service.generateVariants(t.Context(), &observationState{}, rag.Query{Text: "question"}, 1)
	require.NoError(t, err)
	require.Equal(t, []string{"rewrite"}, variants)
	require.Equal(t, service.MultiQueryPrompt, generator.request.Prompt)
}

func TestStrategiesPreserveDeterministicRankingsAndContributions(t *testing.T) {
	service, _, _ := serviceFixture()
	for _, strategy := range []Strategy{StrategyBM25, StrategyVector, StrategyRRF} {
		result, err := service.Retrieve(t.Context(), requestFixture(strategy))
		require.NoError(t, err)
		require.Equal(t, strategy, result.Strategy)
		require.Len(t, result.Evidence, 2)
		require.NotEmpty(t, result.Fused)
		for index, evidence := range result.Evidence {
			require.Equal(t, index+1, evidence.Rank)
		}
	}
	bm25, err := service.Retrieve(t.Context(), requestFixture(StrategyBM25))
	require.NoError(t, err)
	require.Equal(t, []string{"chunk-a", "chunk-b"}, evidenceIDs(bm25.Evidence))
	require.Len(t, bm25.Channels[string(StrategyBM25)], 2)

	rrf, err := service.Retrieve(t.Context(), requestFixture(StrategyRRF))
	require.NoError(t, err)
	require.Equal(t, "chunk-b", rrf.Fused[0].ChunkID)
	require.Len(t, rrf.Fused[0].Contributions, 2)
}

func TestRRFTieBreaksByChunkID(t *testing.T) {
	service, lexical, vector := serviceFixture()
	lexical.hits = []rag.Hit{{RepresentationID: "rep-b", ChunkID: "chunk-b", DocumentID: "doc-b", Rank: 1}}
	vector.hits = []rag.Hit{{RepresentationID: "rep-a", ChunkID: "chunk-a", DocumentID: "doc-a", Rank: 1}}
	result, err := service.Retrieve(t.Context(), requestFixture(StrategyRRF))
	require.NoError(t, err)
	require.Equal(t, []string{"chunk-a", "chunk-b"}, []string{result.Fused[0].ChunkID, result.Fused[1].ChunkID})
}

func TestRerankedStrategyBoundsCandidates(t *testing.T) {
	service, _, _ := serviceFixture()
	reranker := &reversingReranker{}
	service.Reranker = reranker
	result, err := service.Retrieve(t.Context(), requestFixture(StrategyRRFReranked))
	require.NoError(t, err)
	require.Equal(t, 3, reranker.candidates)
	require.Len(t, result.Evidence, 2)
}

func TestRerankedStrategyValidatesIdentitiesAndRehydratesSourceEvidence(t *testing.T) {
	service, _, _ := serviceFixture()
	request := requestFixture(StrategyRRFReranked)
	score := 0.9
	service.Reranker = fixedReranker{evidence: []rag.Evidence{
		{Chunk: rag.Chunk{ID: "chunk-c", Text: "provider replacement"}, RerankerScore: &score},
		{Chunk: rag.Chunk{ID: "chunk-b"}},
	}}
	result, err := service.Retrieve(t.Context(), request)
	require.NoError(t, err)
	require.Equal(t, []string{"chunk-c", "chunk-b"}, evidenceIDs(result.Evidence))
	require.Equal(t, "c", result.Evidence[0].Chunk.Text)
	require.Equal(t, &score, result.Evidence[0].RerankerScore)

	for name, returned := range map[string][]rag.Evidence{
		"too few":   {{Chunk: rag.Chunk{ID: "chunk-c"}}},
		"duplicate": {{Chunk: rag.Chunk{ID: "chunk-c"}}, {Chunk: rag.Chunk{ID: "chunk-c"}}},
		"unknown":   {{Chunk: rag.Chunk{ID: "chunk-c"}}, {Chunk: rag.Chunk{ID: "injected"}}},
		"non-finite score": {
			{Chunk: rag.Chunk{ID: "chunk-c"}, RerankerScore: float64Pointer(math.NaN())},
			{Chunk: rag.Chunk{ID: "chunk-b"}},
		},
	} {
		t.Run(name, func(t *testing.T) {
			service.Reranker = fixedReranker{evidence: returned}
			_, err := service.Retrieve(t.Context(), request)
			require.ErrorContains(t, err, "validate reranker output")
		})
	}
}

func TestQuestionAndRetrievalQueryHaveSeparateRoutes(t *testing.T) {
	service, lexical, _ := serviceFixture()
	request := requestFixture(StrategyBM25)
	request.RetrievalQuery = "retrieval rewrite"
	result, err := service.Retrieve(t.Context(), request)
	require.NoError(t, err)
	prepared, err := service.Prepare(t.Context(), request, result)
	require.NoError(t, err)
	require.Equal(t, "retrieval rewrite", lexical.query.Text)
	require.Equal(t, "question", prepared.GenerationRequest.Text)
}

func TestAnswerObservationsAreOrderedAndComplete(t *testing.T) {
	service, _, _ := serviceFixture()
	generator := &fixedGenerator{result: rag.GenerationResult{
		Text: `{"answer":"A.","citation_chunk_ids":["chunk-a"],"abstained":false}`,
	}}
	service.Generator = generator
	var observations []Observation
	service.Observer = func(_ context.Context, observation Observation) error {
		observations = append(observations, observation)
		return nil
	}
	result, err := service.Answer(t.Context(), requestFixture(StrategyRRF))
	require.NoError(t, err)
	require.True(t, result.Contract.Valid)
	require.Equal(t, result.Context.Evidence, generator.request.Evidence)
	for index, observation := range observations {
		require.Equal(t, index+1, observation.Sequence)
	}
	require.Equal(t, []Stage{
		StageLexical, StageLexical,
		StageVector, StageVector,
		StageFusion, StageFusion,
		StageContext, StageContext,
		StageGeneration, StageGeneration,
		StageContract, StageContract,
	}, observationStages(observations))
}

func TestAnswerStopsAfterProviderFailure(t *testing.T) {
	service, _, _ := serviceFixture()
	service.Generator = &fixedGenerator{err: errors.New("provider unavailable")}
	var observations []Observation
	service.Observer = func(_ context.Context, observation Observation) error {
		observations = append(observations, observation)
		return nil
	}
	result, err := service.Answer(t.Context(), requestFixture(StrategyBM25))
	require.ErrorContains(t, err, "provider unavailable")
	require.Equal(t, AnswerFailureProvider, result.Contract.FailureCategory)
	require.Equal(t, ObservationFailed, observations[len(observations)-1].Status)
	require.NotContains(t, observationStages(observations), StageContract)
}

func TestOrdinalCitationsAreValidatedThenMappedToImmutableChunkIDs(t *testing.T) {
	service, _, _ := serviceFixture()
	generator := &fixedGenerator{result: rag.GenerationResult{
		Text: `{"answer":"A.","citation_chunk_ids":["E1"],"abstained":false}`,
	}}
	service.Generator = generator
	service.CitationStyle = CitationStyleOrdinal
	result, err := service.Answer(t.Context(), requestFixture(StrategyBM25))
	require.NoError(t, err)
	require.True(t, result.Contract.Valid)
	require.Equal(t, "E1", generator.request.Evidence[0].Chunk.ID)
	require.Equal(t, "chunk-a", result.Answer.CitationChunkIDs[0])
	require.Equal(t, "chunk-a", result.CitationLabels["E1"])
}

func TestOrdinalCitationsRejectUnknownLabels(t *testing.T) {
	service, _, _ := serviceFixture()
	service.Generator = &fixedGenerator{result: rag.GenerationResult{
		Text: `{"answer":"A.","citation_chunk_ids":["E9"],"abstained":false}`,
	}}
	service.CitationStyle = CitationStyleOrdinal
	result, err := service.Answer(t.Context(), requestFixture(StrategyBM25))
	require.NoError(t, err)
	require.False(t, result.Contract.Valid)
	require.Equal(t, AnswerFailureContract, result.Contract.FailureCategory)
}

func TestOrdinalCitationsRejectMutatedLabelMapping(t *testing.T) {
	service, _, _ := serviceFixture()
	service.CitationStyle = CitationStyleOrdinal
	request := requestFixture(StrategyBM25)
	retrieved, err := service.Retrieve(t.Context(), request)
	require.NoError(t, err)
	prepared, err := service.Prepare(t.Context(), request, retrieved)
	require.NoError(t, err)
	prepared.CitationLabels["E1"] = "chunk-c"
	_, err = service.Interpret(t.Context(), prepared, rag.GenerationResult{
		Text: `{"answer":"A.","citation_chunk_ids":["E1"],"abstained":false}`,
	})
	require.ErrorContains(t, err, "unauthorized chunk")
}

func TestInterpretUsesPreparedCitationStyle(t *testing.T) {
	service, _, _ := serviceFixture()
	service.CitationStyle = CitationStyleOrdinal
	request := requestFixture(StrategyBM25)
	retrieved, err := service.Retrieve(t.Context(), request)
	require.NoError(t, err)
	prepared, err := service.Prepare(t.Context(), request, retrieved)
	require.NoError(t, err)
	service.CitationStyle = CitationStyleChunkID
	result, err := service.Interpret(t.Context(), prepared, rag.GenerationResult{
		Text: `{"answer":"A.","citation_chunk_ids":["E1"],"abstained":false}`,
	})
	require.NoError(t, err)
	require.True(t, result.Contract.Valid)
	require.Equal(t, []string{"chunk-a"}, result.Answer.CitationChunkIDs)
}

type traceAugmenter struct{}

func (traceAugmenter) Augment(_ context.Context, result RetrievalResult, _ content.Store) (RetrievalResult, json.RawMessage, error) {
	result.Strategy = "augmented"
	return result, json.RawMessage(`{"gate":"closed"}`), nil
}

func TestRetrievePersistsOpaqueAugmentationTrace(t *testing.T) {
	service, _, _ := serviceFixture()
	service.Augmenter = traceAugmenter{}
	result, err := service.Retrieve(t.Context(), requestFixture(StrategyBM25))
	require.NoError(t, err)
	require.Equal(t, Strategy("augmented"), result.Strategy)
	require.JSONEq(t, `{"gate":"closed"}`, string(result.AugmentationTrace))
}

type tamperingAugmenter struct{ evidence []rag.Evidence }

func (a tamperingAugmenter) Augment(_ context.Context, result RetrievalResult, _ content.Store) (RetrievalResult, json.RawMessage, error) {
	result.Evidence = a.evidence
	return result, nil, nil
}

func TestRetrieveRebindsAugmenterEvidenceToOwnedChunks(t *testing.T) {
	service, _, _ := serviceFixture()
	service.Augmenter = tamperingAugmenter{evidence: []rag.Evidence{{Chunk: rag.Chunk{ID: "chunk-a", Text: "tampered"}}}}
	result, err := service.Retrieve(t.Context(), requestFixture(StrategyBM25))
	require.NoError(t, err)
	require.NotEqual(t, "tampered", result.Evidence[0].Chunk.Text)
}

func TestRetrieveRejectsUnknownAugmenterEvidence(t *testing.T) {
	service, _, _ := serviceFixture()
	service.Augmenter = tamperingAugmenter{evidence: []rag.Evidence{{Chunk: rag.Chunk{ID: "unknown"}}}}
	_, err := service.Retrieve(t.Context(), requestFixture(StrategyBM25))
	require.ErrorContains(t, err, "load augmenter chunks")
}

func TestRetrieveRejectsNonFiniteAugmenterScores(t *testing.T) {
	for name, evidence := range map[string]rag.Evidence{
		"retrieval": {Chunk: rag.Chunk{ID: "chunk-a"}, RetrievalScore: math.NaN()},
		"reranker":  {Chunk: rag.Chunk{ID: "chunk-a"}, RerankerScore: float64Pointer(math.Inf(1))},
	} {
		t.Run(name, func(t *testing.T) {
			service, _, _ := serviceFixture()
			service.Augmenter = tamperingAugmenter{evidence: []rag.Evidence{evidence}}
			_, err := service.Retrieve(t.Context(), requestFixture(StrategyBM25))
			require.ErrorContains(t, err, "non-finite")
		})
	}
}

func TestRetrieveRejectsNonFiniteSearchHitScore(t *testing.T) {
	service, lexical, _ := serviceFixture()
	lexical.hits[0].Score = math.NaN()
	_, err := service.Retrieve(t.Context(), requestFixture(StrategyBM25))
	require.ErrorContains(t, err, "validate lexical search results")
}

func TestValidateRequestRejectsUnknownCitationStyle(t *testing.T) {
	service, _, _ := serviceFixture()
	service.CitationStyle = "markdown"
	err := service.ValidateRequest(requestFixture(StrategyBM25))
	require.ErrorContains(t, err, "unsupported citation style")
}

var _ RetrievalAugmenter = traceAugmenter{}

func TestValidateRequestRejectsMissingDependenciesAndInvalidLimits(t *testing.T) {
	service := &Service{}
	err := service.ValidateRequest(requestFixture(StrategyVector))
	require.ErrorContains(t, err, "vector searcher")

	service = &Service{Lexical: &fixedSearcher{}}
	require.ErrorContains(t, service.ValidateRequest(requestFixture(StrategyBM25)), "content store")

	service, _, _ = serviceFixture()
	request := requestFixture(StrategyBM25)
	request.Config.EvidenceK = 0
	require.ErrorContains(t, service.ValidateRequest(request), "evidence K")

	request = requestFixture(StrategyRRFReranked)
	require.ErrorContains(t, service.ValidateRequest(request), "reranker")

	request = requestFixture(StrategyBM25)
	request.Config.QueryVariants = -1
	require.ErrorContains(t, service.ValidateRequest(request), "query variants")
}

func float64Pointer(value float64) *float64 { return &value }

func TestCancellationStopsAtSearch(t *testing.T) {
	service, _, _ := serviceFixture()
	searcher := contextSearcher{}
	service.Lexical = searcher
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := service.Retrieve(ctx, requestFixture(StrategyBM25))
	require.ErrorIs(t, err, context.Canceled)
}

type contextSearcher struct{}

func (contextSearcher) Search(ctx context.Context, _ rag.Query, _ int) ([]rag.Hit, error) {
	return nil, ctx.Err()
}

func evidenceIDs(evidence []rag.Evidence) []string {
	ret := make([]string, len(evidence))
	for i := range evidence {
		ret[i] = evidence[i].Chunk.ID
	}
	return ret
}

func observationStages(observations []Observation) []Stage {
	ret := make([]Stage, len(observations))
	for i := range observations {
		ret[i] = observations[i].Stage
	}
	return ret
}
