package answering

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/go-go-golems/ragkit/rag/content"
	"github.com/go-go-golems/ragkit/rag/retrieval"
	"github.com/pkg/errors"
)

// Service owns the semantic RAG pipeline while callers own interaction,
// persistence, evaluation, and provider configuration.
type Service struct {
	Lexical         rag.Searcher
	Vector          rag.Searcher
	Reranker        rag.Reranker
	Generator       rag.Generator
	Content         content.Store
	RerankingModel  string
	GenerationModel string
	Prompt          string
	OutputSchema    string
	Observer        Observer
	Augmenter       RetrievalAugmenter
	CitationStyle   CitationStyle
	// MultiQueryPrompt and HyDEPrompt override the query-transformation
	// prompts; empty selects the defaults. Prompts are experiment identity:
	// a changed prompt is a new arm and must be recorded in run configs.
	MultiQueryPrompt string
	HyDEPrompt       string
	// ContractKind overrides the generation-request contract identity;
	// empty selects GroundedAnswerKindV1.
	ContractKind string
}

// DefaultMultiQueryPrompt is the fmt format string for the multi-query
// rewrite; both verbs receive the variant count.
const DefaultMultiQueryPrompt = "Rewrite the user's question as %d alternative" +
	" search queries that preserve the question's domain-specific entities" +
	" and intent. Return exactly one JSON object: {\"variants\": [\"...\"]} with" +
	" %d strings and nothing else."

// DefaultHyDEPrompt asks for the hypothetical answer HyDE embeds in place of
// the raw question on the vector channel.
const DefaultHyDEPrompt = "Write a short plausible answer to the question," +
	" two to four sentences, using the terminology and domain of the question." +
	" Plain text only."

// ValidateRequest rejects ambiguous or impossible work before any component is
// called.
func (s *Service) ValidateRequest(request Request) error {
	switch s.CitationStyle {
	case "", CitationStyleChunkID, CitationStyleOrdinal:
	default:
		return errors.Errorf("unsupported citation style %q", s.CitationStyle)
	}
	if strings.TrimSpace(request.TurnID) == "" {
		return errors.New("turn ID is required")
	}
	if strings.TrimSpace(request.Query.ID) == "" {
		return errors.New("query ID is required")
	}
	if strings.TrimSpace(request.Query.Text) == "" {
		return errors.New("question text is required")
	}
	switch request.Config.Strategy {
	case StrategyBM25:
		if s.Lexical == nil {
			return errors.New("bm25 strategy requires a lexical searcher")
		}
	case StrategyVector:
		if s.Vector == nil {
			return errors.New("vector strategy requires a vector searcher")
		}
	case StrategyRRF:
		if s.Lexical == nil || s.Vector == nil {
			return errors.New("rrf strategy requires lexical and vector searchers")
		}
	case StrategyRRFReranked:
		if s.Lexical == nil || s.Vector == nil {
			return errors.New("rrf-reranked strategy requires lexical and vector searchers")
		}
		if s.Reranker == nil {
			return errors.New("rrf-reranked strategy requires a reranker")
		}
	case StrategyMultiQuery:
		if s.Lexical == nil || s.Vector == nil {
			return errors.New("multi-query strategy requires lexical and vector searchers")
		}
		if s.Generator == nil {
			return errors.New("multi-query strategy requires a generator for the reformulations")
		}
	case StrategyHyDE:
		if s.Lexical == nil || s.Vector == nil {
			return errors.New("hyde strategy requires lexical and vector searchers")
		}
		if s.Generator == nil {
			return errors.New("hyde strategy requires a generator for the hypothetical answer")
		}
	default:
		return errors.Errorf("unknown retrieval strategy %q", request.Config.Strategy)
	}
	if s.Content == nil {
		return errors.New("answering requires a content store")
	}
	if request.Config.RetrieveK <= 0 {
		return errors.New("retrieve K must be greater than zero")
	}
	if request.Config.EvidenceK <= 0 {
		return errors.New("evidence K must be greater than zero")
	}
	if request.Config.MaximumContextRunes <= 0 {
		return errors.New("maximum context runes must be greater than zero")
	}
	if request.Config.QueryVariants < 0 {
		return errors.New("query variants must not be negative")
	}
	switch request.Config.Strategy {
	case StrategyRRF, StrategyRRFReranked, StrategyMultiQuery, StrategyHyDE:
		if math.IsNaN(request.Config.RRFConstant) || math.IsInf(request.Config.RRFConstant, 0) || request.Config.RRFConstant <= 0 {
			return errors.New("RRF constant must be finite and greater than zero")
		}
		for channel, weight := range request.Config.RRFWeights {
			if channel != "bm25" && channel != "vector" {
				return errors.Errorf("unknown RRF weight channel %q", channel)
			}
			if math.IsNaN(weight) || math.IsInf(weight, 0) || weight <= 0 {
				return errors.Errorf("RRF weight for channel %q must be finite and greater than zero", channel)
			}
		}
	case StrategyBM25, StrategyVector:
	}
	if request.Config.Strategy == StrategyRRFReranked &&
		request.Config.RerankCandidates < request.Config.EvidenceK {
		return errors.New("rerank candidates must be at least evidence K")
	}
	return nil
}

// Retrieve executes the selected retrieval strategy and retains each
// intermediate ranking.
func (s *Service) Retrieve(ctx context.Context, request Request) (RetrievalResult, error) {
	if err := s.ValidateRequest(request); err != nil {
		return RetrievalResult{}, err
	}
	state := observationState{turnID: request.TurnID, observer: s.Observer}
	return s.retrieve(ctx, request, &state)
}

// Prepare applies context admission and constructs the exact provider request.
func (s *Service) Prepare(ctx context.Context, request Request, result RetrievalResult) (Prepared, error) {
	state := observationState{turnID: request.TurnID, observer: s.Observer}
	return s.prepare(ctx, request, result, &state)
}

// Interpret applies strict answer parsing and grounding validation to a
// provider result.
func (s *Service) Interpret(ctx context.Context, prepared Prepared, raw rag.GenerationResult) (Result, error) {
	state := observationState{turnID: prepared.Request.TurnID, observer: s.Observer}
	return s.interpret(ctx, prepared, raw, &state)
}

// Answer executes one complete answering operation.
func (s *Service) Answer(ctx context.Context, request Request) (Result, error) {
	started := time.Now()
	if err := s.ValidateRequest(request); err != nil {
		return Result{}, err
	}
	if s.Generator == nil {
		return Result{}, errors.New("answering requires a generator")
	}
	state := observationState{turnID: request.TurnID, observer: s.Observer}
	retrieved, err := s.retrieve(ctx, request, &state)
	if err != nil {
		return Result{}, err
	}
	prepared, err := s.prepare(ctx, request, retrieved, &state)
	if err != nil {
		return Result{}, err
	}
	stageStarted := time.Now()
	if err := state.emit(ctx, StageGeneration, ObservationStarted, stageStarted, 0, nil, ""); err != nil {
		return Result{}, err
	}
	raw, err := s.Generator.Generate(ctx, prepared.GenerationRequest)
	if err != nil {
		contract := AnswerContractResult{
			FailureCategory: AnswerFailureProvider,
			Failures:        []string{err.Error()},
		}
		_ = state.emit(context.WithoutCancel(ctx), StageGeneration, ObservationFailed, stageStarted, time.Since(stageStarted), contract, err.Error())
		return Result{Prepared: prepared, Raw: raw, Contract: contract, StartedAt: started.UTC(), Duration: time.Since(started)},
			errors.Wrap(err, "generate grounded answer")
	}
	if err := state.emit(ctx, StageGeneration, ObservationCompleted, stageStarted, time.Since(stageStarted), raw, ""); err != nil {
		return Result{Prepared: prepared, Raw: raw, StartedAt: started.UTC(), Duration: time.Since(started)}, err
	}
	result, err := s.interpret(ctx, prepared, raw, &state)
	result.StartedAt = started.UTC()
	result.Duration = time.Since(started)
	return result, err
}

func (s *Service) retrieve(ctx context.Context, request Request, state *observationState) (RetrievalResult, error) {
	started := time.Now()
	query := request.Query
	if strings.TrimSpace(request.RetrievalQuery) != "" {
		query.Text = request.RetrievalQuery
	}
	result := RetrievalResult{
		Strategy:  request.Config.Strategy,
		Query:     query,
		Channels:  make(map[string][]rag.Hit),
		StartedAt: started.UTC(),
	}
	switch request.Config.Strategy {
	case StrategyBM25:
		hits, err := s.search(ctx, state, StageLexical, s.Lexical, query, request.Config.RetrieveK)
		if err != nil {
			return RetrievalResult{}, err
		}
		result.Channels[string(StrategyBM25)] = hits
		result.Fused = retrieval.FromHits(hits)
	case StrategyVector:
		hits, err := s.search(ctx, state, StageVector, s.Vector, query, request.Config.RetrieveK)
		if err != nil {
			return RetrievalResult{}, err
		}
		result.Channels[string(StrategyVector)] = hits
		result.Fused = retrieval.FromHits(hits)
	case StrategyRRF, StrategyRRFReranked:
		lexical, err := s.search(ctx, state, StageLexical, s.Lexical, query, request.Config.RetrieveK)
		if err != nil {
			return RetrievalResult{}, err
		}
		vector, err := s.search(ctx, state, StageVector, s.Vector, query, request.Config.RetrieveK)
		if err != nil {
			return RetrievalResult{}, err
		}
		result.Channels[string(StrategyBM25)] = lexical
		result.Channels[string(StrategyVector)] = vector
		fusionStarted := time.Now()
		if err := state.emit(ctx, StageFusion, ObservationStarted, fusionStarted, 0, nil, ""); err != nil {
			return RetrievalResult{}, err
		}
		weights := request.Config.RRFWeights
		if len(weights) == 0 {
			weights = map[string]float64{
				string(StrategyBM25):   1,
				string(StrategyVector): 1,
			}
		}
		result.Fused, err = retrieval.WeightedRRF(result.Channels, retrieval.RRFConfig{
			RankConstant: request.Config.RRFConstant,
			Weights:      weights,
		})
		if err != nil {
			_ = state.emit(context.WithoutCancel(ctx), StageFusion, ObservationFailed, fusionStarted, time.Since(fusionStarted), nil, err.Error())
			return RetrievalResult{}, errors.Wrap(err, "fuse retrieval channels")
		}
		if err := state.emit(ctx, StageFusion, ObservationCompleted, fusionStarted, time.Since(fusionStarted), result.Fused, ""); err != nil {
			return RetrievalResult{}, err
		}
	case StrategyMultiQuery:
		variants, variantErr, usage, err := s.generateVariants(ctx, state, query, variantCount(request.Config))
		if err != nil {
			return RetrievalResult{}, err
		}
		result.Variants, result.VariantError, result.QueryGenerationUsage = variants, variantErr, usage
		// The fallback rule: reformulation failure degrades to the plain
		// question, never to a failed turn. The record says what degraded.
		queries := append([]rag.Query{query}, variantQueries(query, variants)...)
		for index, variant := range queries {
			suffix := fmt.Sprintf(":q%d", index)
			lexical, err := s.search(ctx, state, StageLexical, s.Lexical, variant, request.Config.RetrieveK)
			if err != nil {
				return RetrievalResult{}, err
			}
			vector, err := s.search(ctx, state, StageVector, s.Vector, variant, request.Config.RetrieveK)
			if err != nil {
				return RetrievalResult{}, err
			}
			result.Channels[string(StrategyBM25)+suffix] = lexical
			result.Channels[string(StrategyVector)+suffix] = vector
		}
		fused, err := s.fuseChannels(ctx, state, result.Channels, request.Config)
		if err != nil {
			return RetrievalResult{}, err
		}
		result.Fused = fused
	case StrategyHyDE:
		hypothetical, variantErr, usage, err := s.generateHypothetical(ctx, state, query)
		if err != nil {
			return RetrievalResult{}, err
		}
		result.VariantError, result.QueryGenerationUsage = variantErr, usage
		vectorQuery := query
		if hypothetical != "" {
			result.Variants = []string{hypothetical}
			vectorQuery.Text = hypothetical
		}
		lexical, err := s.search(ctx, state, StageLexical, s.Lexical, query, request.Config.RetrieveK)
		if err != nil {
			return RetrievalResult{}, err
		}
		vector, err := s.search(ctx, state, StageVector, s.Vector, vectorQuery, request.Config.RetrieveK)
		if err != nil {
			return RetrievalResult{}, err
		}
		result.Channels[string(StrategyBM25)] = lexical
		result.Channels[string(StrategyVector)] = vector
		fused, err := s.fuseChannels(ctx, state, result.Channels, request.Config)
		if err != nil {
			return RetrievalResult{}, err
		}
		result.Fused = fused
	}
	evidenceLimit := request.Config.EvidenceK
	if request.Config.Strategy == StrategyRRFReranked {
		evidenceLimit = request.Config.RerankCandidates
	}
	evidence, err := retrieval.HydrateFromStore(ctx, result.Fused, s.Content, evidenceLimit)
	if err != nil {
		return RetrievalResult{}, errors.Wrap(err, "hydrate retrieval evidence")
	}
	if request.Config.Strategy == StrategyRRFReranked {
		rerankStarted := time.Now()
		if err := state.emit(ctx, StageReranking, ObservationStarted, rerankStarted, 0, evidence, ""); err != nil {
			return RetrievalResult{}, err
		}
		reranked, rerankErr := s.Reranker.Rerank(ctx, rag.RerankRequest{
			Model:      s.RerankingModel,
			Query:      query,
			Candidates: evidence,
			Results:    request.Config.EvidenceK,
		})
		if rerankErr != nil {
			_ = state.emit(context.WithoutCancel(ctx), StageReranking, ObservationFailed, rerankStarted, time.Since(rerankStarted), nil, rerankErr.Error())
			return RetrievalResult{}, errors.Wrap(rerankErr, "rerank RRF candidates")
		}
		evidence, err = validateRerankedEvidence(evidence, reranked.Evidence, request.Config.EvidenceK)
		if err != nil {
			_ = state.emit(context.WithoutCancel(ctx), StageReranking, ObservationFailed, rerankStarted, time.Since(rerankStarted), nil, err.Error())
			return RetrievalResult{}, errors.Wrap(err, "validate reranker output")
		}
		reranked.Evidence = evidence
		if err := state.emit(ctx, StageReranking, ObservationCompleted, rerankStarted, time.Since(rerankStarted), reranked, ""); err != nil {
			return RetrievalResult{}, err
		}
	}
	result.Evidence = evidence
	if s.Augmenter != nil {
		augmented, trace, err := s.Augmenter.Augment(ctx, result, s.Content)
		if err != nil {
			return RetrievalResult{}, errors.Wrap(err, "augment retrieval")
		}
		augmented.Evidence, err = validateAugmentedEvidence(ctx, s.Content, augmented.Evidence)
		if err != nil {
			return RetrievalResult{}, errors.Wrap(err, "validate augmenter output")
		}
		result = augmented
		result.AugmentationTrace = trace
	}
	result.Duration = time.Since(started)
	return result, nil
}

func (s *Service) search(
	ctx context.Context,
	state *observationState,
	stage Stage,
	searcher rag.Searcher,
	query rag.Query,
	limit int,
) ([]rag.Hit, error) {
	started := time.Now()
	if err := state.emit(ctx, stage, ObservationStarted, started, 0, query, ""); err != nil {
		return nil, err
	}
	hits, err := searcher.Search(ctx, query, limit)
	if err != nil {
		_ = state.emit(context.WithoutCancel(ctx), stage, ObservationFailed, started, time.Since(started), nil, err.Error())
		return nil, errors.Wrapf(err, "execute %s search", stage)
	}
	if err := rag.ValidateHits(hits); err != nil {
		_ = state.emit(context.WithoutCancel(ctx), stage, ObservationFailed, started, time.Since(started), nil, err.Error())
		return nil, errors.Wrapf(err, "validate %s search results", stage)
	}
	collapsed, err := retrieval.Collapse(hits, retrieval.TargetChunk)
	if err != nil {
		_ = state.emit(context.WithoutCancel(ctx), stage, ObservationFailed, started, time.Since(started), nil, err.Error())
		return nil, errors.Wrapf(err, "collapse %s search", stage)
	}
	if err := state.emit(ctx, stage, ObservationCompleted, started, time.Since(started), collapsed, ""); err != nil {
		return nil, err
	}
	return collapsed, nil
}

func (s *Service) prepare(ctx context.Context, request Request, retrieved RetrievalResult, state *observationState) (Prepared, error) {
	started := time.Now()
	if err := state.emit(ctx, StageContext, ObservationStarted, started, 0, nil, ""); err != nil {
		return Prepared{}, err
	}
	contextResult := ApplyContextPolicy(
		retrieved.Evidence,
		request.Config.EvidenceK,
		request.Config.MaximumContextRunes,
	)
	prepared := Prepared{
		Request:       request,
		Retrieval:     retrieved,
		Context:       contextResult,
		CitationStyle: s.CitationStyle,
		GenerationRequest: rag.GenerationRequest{
			Kind:         s.contractKind(),
			Model:        s.GenerationModel,
			Prompt:       s.Prompt,
			Text:         request.Query.Text,
			Evidence:     contextResult.Evidence,
			OutputSchema: s.OutputSchema,
		},
	}
	if prepared.CitationStyle == CitationStyleOrdinal {
		prepared.CitationLabels = make(map[string]string, len(prepared.Context.Evidence))
		for index := range prepared.Context.Evidence {
			label := fmt.Sprintf("E%d", index+1)
			prepared.CitationLabels[label] = prepared.Context.Evidence[index].Chunk.ID
			prepared.Context.Evidence[index].Chunk.ID = label
		}
		prepared.GenerationRequest.Evidence = append(
			[]rag.Evidence(nil), prepared.Context.Evidence...,
		)
	}
	if err := state.emit(ctx, StageContext, ObservationCompleted, started, time.Since(started), contextResult, ""); err != nil {
		return Prepared{}, err
	}
	return prepared, nil
}

func (s *Service) interpret(ctx context.Context, prepared Prepared, raw rag.GenerationResult, state *observationState) (Result, error) {
	started := time.Now()
	if err := state.emit(ctx, StageContract, ObservationStarted, started, 0, nil, ""); err != nil {
		return Result{}, err
	}
	validatedPrepared, err := s.validateStagedPrepared(ctx, prepared)
	if err != nil {
		return Result{}, err
	}
	prepared = validatedPrepared
	answer, contract := ParseGroundedAnswer(raw.Text, prepared.Context.Evidence)
	if contract.Valid && prepared.CitationStyle == CitationStyleOrdinal {
		for index, label := range answer.CitationChunkIDs {
			chunkID, ok := prepared.CitationLabels[label]
			if !ok {
				return Result{}, errors.Errorf("validated citation label %q has no immutable chunk mapping", label)
			}
			answer.CitationChunkIDs[index] = chunkID
		}
	}
	result := Result{
		Prepared: prepared,
		Raw:      raw,
		Answer:   answer,
		Contract: contract,
	}
	if err := state.emit(ctx, StageContract, ObservationCompleted, started, time.Since(started), contract, ""); err != nil {
		return Result{}, err
	}
	return result, nil
}

type observationState struct {
	turnID   string
	sequence int
	observer Observer
}

func (s *observationState) emit(
	ctx context.Context,
	stage Stage,
	status ObservationStatus,
	started time.Time,
	duration time.Duration,
	detail any,
	errText string,
) error {
	if s.observer == nil {
		return nil
	}
	s.sequence++
	err := s.observer(ctx, Observation{
		TurnID:   s.turnID,
		Sequence: s.sequence,
		Stage:    stage,
		Status:   status,
		At:       time.Now().UTC(),
		Duration: duration,
		Error:    errText,
		Detail:   detail,
	})
	return errors.Wrap(err, "record answering observation")
}

// variantCount applies the multi-query default: three reformulations.
func variantCount(config RetrievalConfig) int {
	if config.QueryVariants > 0 {
		return config.QueryVariants
	}
	return 3
}

// variantQueries mints identities for the reformulations. Each variant keeps
// the source query id with a suffix, so downstream joins still reach the
// original question.
func variantQueries(source rag.Query, variants []string) []rag.Query {
	queries := make([]rag.Query, 0, len(variants))
	for index, text := range variants {
		queries = append(queries, rag.Query{
			ID: fmt.Sprintf("%s-v%d", source.ID, index+1), Text: text,
		})
	}
	return queries
}

// fuseChannels runs weighted RRF over however many channels retrieval
// produced. Channel weights fall back per prefix: a "bm25:q2" channel takes
// the configured bm25 weight, so the config's two knobs govern every variant.
func (s *Service) fuseChannels(
	ctx context.Context,
	state *observationState,
	channels map[string][]rag.Hit,
	config RetrievalConfig,
) ([]rag.FusedHit, error) {
	fusionStarted := time.Now()
	if err := state.emit(ctx, StageFusion, ObservationStarted, fusionStarted, 0, nil, ""); err != nil {
		return nil, err
	}
	weights := map[string]float64{}
	for name := range channels {
		base := name
		if index := strings.IndexByte(name, ':'); index >= 0 {
			base = name[:index]
		}
		if weight, ok := config.RRFWeights[base]; ok && weight > 0 {
			weights[name] = weight
		}
	}
	fused, err := retrieval.WeightedRRF(channels, retrieval.RRFConfig{
		RankConstant: config.RRFConstant, Weights: weights,
	})
	if err != nil {
		_ = state.emit(context.WithoutCancel(ctx), StageFusion, ObservationFailed, fusionStarted, time.Since(fusionStarted), nil, err.Error())
		return nil, errors.Wrap(err, "fuse retrieval channels")
	}
	if err := state.emit(ctx, StageFusion, ObservationCompleted, fusionStarted, time.Since(fusionStarted), fused, ""); err != nil {
		return nil, err
	}
	return fused, nil
}

// generateVariants asks the generator for query reformulations under a strict
// JSON contract. Generation and contract failures are reported as fallback
// details; observer failures remain operational errors and stop the turn.
func (s *Service) generateVariants(
	ctx context.Context,
	state *observationState,
	query rag.Query,
	count int,
) ([]string, string, rag.Usage, error) {
	started := time.Now()
	if err := state.emit(ctx, StageQueryGen, ObservationStarted, started, 0, nil, ""); err != nil {
		return nil, "", rag.Usage{}, err
	}
	prompt := s.MultiQueryPrompt
	if prompt == "" {
		prompt = fmt.Sprintf(DefaultMultiQueryPrompt, count, count)
	}
	raw, err := s.Generator.Generate(ctx, rag.GenerationRequest{
		Kind: "ttc-query-variants-v1", Model: s.GenerationModel,
		Prompt: prompt, Text: query.Text,
	})
	if err != nil {
		message := err.Error()
		if observeErr := state.emit(context.WithoutCancel(ctx), StageQueryGen, ObservationFailed, started, time.Since(started), nil, message); observeErr != nil {
			return nil, message, raw.Usage, observeErr
		}
		return nil, message, raw.Usage, nil
	}
	var parsed struct {
		Variants []string `json:"variants"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw.Text)), &parsed); err != nil {
		message := "decode variants: " + err.Error()
		if observeErr := state.emit(context.WithoutCancel(ctx), StageQueryGen, ObservationFailed, started, time.Since(started), raw, message); observeErr != nil {
			return nil, message, raw.Usage, observeErr
		}
		return nil, message, raw.Usage, nil
	}
	variants := make([]string, 0, count)
	seen := map[string]bool{strings.TrimSpace(query.Text): true}
	for _, variant := range parsed.Variants {
		variant = strings.TrimSpace(variant)
		if variant == "" || seen[variant] {
			continue
		}
		seen[variant] = true
		variants = append(variants, variant)
		if len(variants) == count {
			break
		}
	}
	if len(variants) != count {
		message := fmt.Sprintf("the generator returned %d distinct usable variants; expected %d", len(variants), count)
		if observeErr := state.emit(context.WithoutCancel(ctx), StageQueryGen, ObservationFailed, started, time.Since(started), raw, message); observeErr != nil {
			return nil, message, raw.Usage, observeErr
		}
		return nil, message, raw.Usage, nil
	}
	if err := state.emit(ctx, StageQueryGen, ObservationCompleted, started, time.Since(started), variants, ""); err != nil {
		return nil, "", raw.Usage, err
	}
	return variants, "", raw.Usage, nil
}

// validateRerankedEvidence treats provider output as an ordering and score
// decision, never as authoritative source evidence. Every returned ID must be
// a distinct candidate, and the provider must return the requested count.
func validateRerankedEvidence(candidates, returned []rag.Evidence, requested int) ([]rag.Evidence, error) {
	want := min(requested, len(candidates))
	if len(returned) != want {
		return nil, errors.Errorf("reranker returned %d results, want %d", len(returned), want)
	}
	byID := make(map[string]rag.Evidence, len(candidates))
	for _, candidate := range candidates {
		byID[candidate.Chunk.ID] = candidate
	}
	seen := make(map[string]bool, len(returned))
	validated := make([]rag.Evidence, 0, len(returned))
	for index, item := range returned {
		id := item.Chunk.ID
		candidate, ok := byID[id]
		if !ok {
			return nil, errors.Errorf("reranker result %d references unknown chunk %q", index, id)
		}
		if seen[id] {
			return nil, errors.Errorf("reranker returned duplicate chunk %q", id)
		}
		seen[id] = true
		if err := validateEvidenceScores("reranker", index, item); err != nil {
			return nil, err
		}
		candidate.Rank = index + 1
		candidate.RerankerScore = item.RerankerScore
		validated = append(validated, candidate)
	}
	return validated, nil
}

// validateAugmentedEvidence treats augmenter chunks as references into the
// service-owned content store. The augmenter may choose order and scores, but
// it may neither introduce nor alter source content.
func validateAugmentedEvidence(ctx context.Context, store content.Store, returned []rag.Evidence) ([]rag.Evidence, error) {
	ids := make([]string, 0, len(returned))
	seen := make(map[string]struct{}, len(returned))
	for index, item := range returned {
		id := item.Chunk.ID
		if _, duplicate := seen[id]; duplicate {
			return nil, errors.Errorf("augmenter returned duplicate chunk %q", id)
		}
		seen[id] = struct{}{}
		if err := validateEvidenceScores("augmenter", index, item); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	byID, err := loadReferencedChunks(ctx, store, ids, "augmenter")
	if err != nil {
		return nil, err
	}
	validated := make([]rag.Evidence, len(returned))
	for index, item := range returned {
		item.Chunk = byID[item.Chunk.ID]
		validated[index] = item
	}
	return validated, nil
}

func validateEvidenceScores(role string, index int, item rag.Evidence) error {
	if math.IsNaN(item.RetrievalScore) || math.IsInf(item.RetrievalScore, 0) {
		return errors.Errorf("%s result %d for chunk %q has a non-finite retrieval score", role, index, item.Chunk.ID)
	}
	if score := item.RerankerScore; score != nil && (math.IsNaN(*score) || math.IsInf(*score, 0)) {
		return errors.Errorf("%s result %d for chunk %q has a non-finite reranker score", role, index, item.Chunk.ID)
	}
	return nil
}

func (s *Service) validateStagedPrepared(ctx context.Context, prepared Prepared) (Prepared, error) {
	if prepared.CitationStyle == CitationStyleOrdinal {
		if err := validateCitationLabels(prepared); err != nil {
			return Prepared{}, err
		}
	}
	validated := append([]rag.Evidence(nil), prepared.Context.Evidence...)
	ids := make([]string, 0, len(validated))
	seen := make(map[string]struct{}, len(validated))
	for index := range validated {
		id := validated[index].Chunk.ID
		if prepared.CitationStyle == CitationStyleOrdinal {
			var ok bool
			id, ok = prepared.CitationLabels[id]
			if !ok {
				return Prepared{}, errors.Errorf("citation label %q has no immutable chunk mapping", validated[index].Chunk.ID)
			}
		}
		if _, duplicate := seen[id]; duplicate {
			return Prepared{}, errors.Errorf("staged context duplicates service chunk %q", id)
		}
		seen[id] = struct{}{}
		ids = append(ids, id)
	}
	byID, err := loadReferencedChunks(ctx, s.Content, ids, "staged context")
	if err != nil {
		return Prepared{}, err
	}
	for index, id := range ids {
		validated[index].Chunk = byID[id]
		if prepared.CitationStyle == CitationStyleOrdinal {
			validated[index].Chunk.ID = fmt.Sprintf("E%d", index+1)
			if prepared.CitationLabels[validated[index].Chunk.ID] != id {
				return Prepared{}, errors.Errorf("citation label %q maps to unauthorized chunk %q", validated[index].Chunk.ID, id)
			}
		}
	}
	prepared.Context.Evidence = validated
	return prepared, nil
}

func loadReferencedChunks(ctx context.Context, store content.Store, ids []string, role string) (map[string]rag.Chunk, error) {
	if store == nil {
		return nil, errors.Errorf("%s requires a content store", role)
	}
	if len(ids) == 0 {
		return map[string]rag.Chunk{}, nil
	}
	chunks, err := content.LoadChunks(ctx, store, ids)
	if err != nil {
		return nil, errors.Wrapf(err, "load %s chunks", role)
	}
	requested := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		requested[id] = struct{}{}
	}
	byID := make(map[string]rag.Chunk, len(chunks))
	for _, chunk := range chunks {
		if _, ok := requested[chunk.ID]; !ok {
			return nil, errors.Errorf("content store returned unrequested %s chunk %q", role, chunk.ID)
		}
		if _, duplicate := byID[chunk.ID]; duplicate {
			return nil, errors.Errorf("content store returned duplicate %s chunk %q", role, chunk.ID)
		}
		byID[chunk.ID] = chunk
	}
	for _, id := range ids {
		if _, ok := byID[id]; !ok {
			return nil, errors.Errorf("%s references unknown service chunk %q", role, id)
		}
	}
	return byID, nil
}

func validateCitationLabels(prepared Prepared) error {
	authorized := ApplyContextPolicy(
		prepared.Retrieval.Evidence,
		prepared.Request.Config.EvidenceK,
		prepared.Request.Config.MaximumContextRunes,
	).Evidence
	if len(prepared.CitationLabels) != len(authorized) {
		return errors.Errorf("citation label mapping has %d entries, want %d", len(prepared.CitationLabels), len(authorized))
	}
	for index, evidence := range authorized {
		label := fmt.Sprintf("E%d", index+1)
		if chunkID := prepared.CitationLabels[label]; chunkID != evidence.Chunk.ID {
			return errors.Errorf("citation label %q maps to unauthorized chunk %q", label, chunkID)
		}
	}
	return nil
}

// generateHypothetical asks the generator for a hypothetical answer whose
// embedding stands in for the question on the vector channel.
func (s *Service) generateHypothetical(
	ctx context.Context,
	state *observationState,
	query rag.Query,
) (string, string, rag.Usage, error) {
	started := time.Now()
	if err := state.emit(ctx, StageQueryGen, ObservationStarted, started, 0, nil, ""); err != nil {
		return "", "", rag.Usage{}, err
	}
	hydePrompt := s.HyDEPrompt
	if hydePrompt == "" {
		hydePrompt = DefaultHyDEPrompt
	}
	raw, err := s.Generator.Generate(ctx, rag.GenerationRequest{
		Kind: "ttc-hypothetical-answer-v1", Model: s.GenerationModel,
		Prompt: hydePrompt,
		Text:   query.Text,
	})
	if err != nil {
		message := err.Error()
		if observeErr := state.emit(context.WithoutCancel(ctx), StageQueryGen, ObservationFailed, started, time.Since(started), nil, message); observeErr != nil {
			return "", message, raw.Usage, observeErr
		}
		return "", message, raw.Usage, nil
	}
	hypothetical := strings.TrimSpace(raw.Text)
	if hypothetical == "" {
		message := "the generator returned an empty hypothetical answer"
		if observeErr := state.emit(context.WithoutCancel(ctx), StageQueryGen, ObservationFailed, started, time.Since(started), raw, message); observeErr != nil {
			return "", message, raw.Usage, observeErr
		}
		return "", message, raw.Usage, nil
	}
	if err := state.emit(ctx, StageQueryGen, ObservationCompleted, started, time.Since(started), hypothetical, ""); err != nil {
		return "", "", raw.Usage, err
	}
	return hypothetical, "", raw.Usage, nil
}

func (s *Service) contractKind() string {
	if s.ContractKind != "" {
		return s.ContractKind
	}
	return GroundedAnswerKindV1
}
