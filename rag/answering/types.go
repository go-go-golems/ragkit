// Package answering composes retrieval, context construction, generation, and
// grounded-answer validation into one observable RAG operation.
package answering

import (
	"context"
	"encoding/json"
	"time"

	"github.com/go-go-golems/ragkit/rag"
)

const (
	StrategyBM25        Strategy = "bm25"
	StrategyVector      Strategy = "vector"
	StrategyRRF         Strategy = "rrf"
	StrategyRRFReranked Strategy = "rrf-reranked"
	// StrategyMultiQuery generates query variants, retrieves each variant on
	// both channels, and fuses every channel with RRF.
	StrategyMultiQuery Strategy = "multi-query"
	// StrategyHyDE generates a hypothetical answer and uses its embedding for
	// the vector channel; the lexical channel keeps the question.
	StrategyHyDE Strategy = "hyde"

	ContextPolicyWholeEvidenceV1 = "whole-evidence-chunks-v1"
	GroundedAnswerKindV1         = "ttc-grounded-answer-v1"
)

// Strategy selects the retrieval and optional reranking operations used for an
// answer.
type Strategy string

// RetrievalConfig contains the per-request retrieval and context controls.
type RetrievalConfig struct {
	Strategy            Strategy           `json:"strategy"`
	RetrieveK           int                `json:"retrieve_k"`
	EvidenceK           int                `json:"evidence_k"`
	RerankCandidates    int                `json:"rerank_candidates,omitempty"`
	RRFConstant         float64            `json:"rrf_constant,omitempty"`
	RRFWeights          map[string]float64 `json:"rrf_weights,omitempty"`
	MaximumContextRunes int                `json:"maximum_context_runes"`
	// QueryVariants is how many reformulations multi-query generates. Zero
	// means the default of three.
	QueryVariants int `json:"query_variants,omitempty"`
}

// Request is one independent, reproducible answer operation. RetrievalQuery is
// allowed to differ from Query.Text; an empty value explicitly means use the
// question text.
type Request struct {
	TurnID         string          `json:"turn_id"`
	ComparisonID   string          `json:"comparison_id,omitempty"`
	ParentTurnID   string          `json:"parent_turn_id,omitempty"`
	Query          rag.Query       `json:"query"`
	RetrievalQuery string          `json:"retrieval_query,omitempty"`
	Config         RetrievalConfig `json:"config"`
}

// RetrievalResult retains channel-local hits, fused rankings, and hydrated
// source evidence as distinct inspection layers.
type RetrievalResult struct {
	Strategy Strategy             `json:"strategy"`
	Query    rag.Query            `json:"query"`
	Disabled bool                 `json:"disabled,omitempty"`
	Reason   string               `json:"reason,omitempty"`
	Channels map[string][]rag.Hit `json:"channels,omitempty"`
	// Variants records every generated reformulation (or the hypothetical
	// answer) verbatim, so a result is reproducible from its record.
	Variants []string `json:"variants,omitempty"`
	// VariantError states why query generation fell back to the plain
	// question. The turn still answers; the record says what degraded.
	VariantError string         `json:"variant_error,omitempty"`
	Fused        []rag.FusedHit `json:"fused,omitempty"`
	Evidence     []rag.Evidence `json:"evidence,omitempty"`
	// AugmentationTrace is an opaque, durable trace supplied by an optional
	// retrieval augmenter. Raw JSON keeps answering independent of the concrete
	// connected-retrieval package while preserving the trace in session records.
	AugmentationTrace json.RawMessage `json:"augmentation_trace,omitempty"`
	StartedAt         time.Time       `json:"started_at"`
	Duration          time.Duration   `json:"duration"`
}

// ContextPolicyResult records the deterministic admission decision that turns
// ranked evidence into the exact evidence supplied to generation.
type ContextPolicyResult struct {
	Evidence        []rag.Evidence `json:"evidence"`
	OmittedChunkIDs []string       `json:"omitted_chunk_ids,omitempty"`
	UsedRunes       int            `json:"used_runes"`
	MaxRunes        int            `json:"max_runes"`
	MaxEvidence     int            `json:"max_evidence"`
	Policy          string         `json:"policy"`
}

// GroundedAnswer is the validated answer exposed to callers.
type GroundedAnswer struct {
	Text             string   `json:"answer"`
	CitationChunkIDs []string `json:"citation_chunk_ids"`
	Abstained        bool     `json:"abstained"`
}

// AnswerFailureCategory identifies the boundary at which provider output
// failed to become a grounded answer.
type AnswerFailureCategory string

const (
	AnswerFailureNone     AnswerFailureCategory = ""
	AnswerFailureParse    AnswerFailureCategory = "parse"
	AnswerFailureContract AnswerFailureCategory = "contract"
	AnswerFailureProvider AnswerFailureCategory = "provider"
)

// AnswerContractResult records deterministic parsing and grounding checks. It
// is not an answer-quality score.
type AnswerContractResult struct {
	Valid           bool                  `json:"valid"`
	FailureCategory AnswerFailureCategory `json:"failure_category,omitempty"`
	Failures        []string              `json:"failures,omitempty"`
}

// Prepared contains the provider request and all deterministic work preceding
// generation. Batch callers may prepare many requests and retain their own
// bounded/cache-aware provider execution.
type Prepared struct {
	Request           Request               `json:"request"`
	Retrieval         RetrievalResult       `json:"retrieval"`
	Context           ContextPolicyResult   `json:"context"`
	GenerationRequest rag.GenerationRequest `json:"generation_request"`
	// CitationLabels maps model-facing evidence labels back to immutable chunk
	// IDs. It is populated only for ordinal citation presentation.
	CitationLabels map[string]string `json:"citation_labels,omitempty"`
}

// RetrievalAugmenter transforms a completed baseline retrieval result before
// answer preparation. Implementations must preserve baseline results when
// their admission gate is closed.
type RetrievalAugmenter interface {
	Augment(context.Context, RetrievalResult, []rag.Chunk) (RetrievalResult, json.RawMessage, error)
}

// CitationStyle controls only the IDs presented to the generation model.
// Returned GroundedAnswer citations always use immutable chunk IDs.
type CitationStyle string

const (
	CitationStyleChunkID CitationStyle = "chunk-id"
	CitationStyleOrdinal CitationStyle = "ordinal"
)

// Result is the complete inspectable answer operation.
type Result struct {
	Prepared
	Raw       rag.GenerationResult `json:"raw_generation"`
	Answer    GroundedAnswer       `json:"answer"`
	Contract  AnswerContractResult `json:"contract"`
	StartedAt time.Time            `json:"started_at"`
	Duration  time.Duration        `json:"duration"`
}

// Stage names one observable pipeline boundary.
type Stage string

const (
	StageQueryGen   Stage = "query-generation"
	StageLexical    Stage = "lexical"
	StageVector     Stage = "vector"
	StageFusion     Stage = "fusion"
	StageReranking  Stage = "reranking"
	StageContext    Stage = "context"
	StageGeneration Stage = "generation"
	StageContract   Stage = "contract"
)

// ObservationStatus records the lifecycle of one stage.
type ObservationStatus string

const (
	ObservationStarted   ObservationStatus = "started"
	ObservationCompleted ObservationStatus = "completed"
	ObservationFailed    ObservationStatus = "failed"
)

// Observation is append-safe and ordered within one Answer call.
type Observation struct {
	TurnID   string            `json:"turn_id"`
	Sequence int               `json:"sequence"`
	Stage    Stage             `json:"stage"`
	Status   ObservationStatus `json:"status"`
	At       time.Time         `json:"at"`
	Duration time.Duration     `json:"duration,omitempty"`
	Error    string            `json:"error,omitempty"`
	Detail   any               `json:"detail,omitempty"`
}

// Observer persists or presents pipeline observations. Returning an error
// stops the answer because an unrecorded stage must not be presented as
// durable.
type Observer func(context.Context, Observation) error
