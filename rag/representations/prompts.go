package representations

// PromptSet carries the generation prompts and cache-identity strings used
// when building generated representations (summaries, questions, contextual
// blurbs, entity expansions, atomic statements, LLM chunk boundaries).
//
// Prompts are cache identity: the generation cache keys on (kind, model,
// prompt digest, text digest, adapter version, context policy), so two
// callers sharing a PromptSet share every cached generation, and a changed
// prompt is a new cache population. Never edit a prompt in place inside a
// long-lived corpus pipeline — introduce a new, versioned PromptSet.
//
// Callers targeting a specific domain construct and own their own PromptSet.
// Ragkit deliberately contains no product prompt text.
type PromptSet struct {
	// AdapterVersion and ContextPolicy identify how a representation
	// generation call is assembled from its inputs.
	AdapterVersion string
	ContextPolicy  string

	// Per-chunk prompts.
	Contextual      string
	Summary2Sent    string
	SummaryKeywords string
	Questions       string
	Entities        string
	Statements      string

	// Document-level prompt (raptor-lite style summary indexing).
	DocumentSummary string

	// Batched variants: one call covering a numbered list of chunks, with
	// per-chunk repair calls for missing entries handled by the caller.
	ContextualBatched      string
	Summary2SentBatched    string
	SummaryKeywordsBatched string
	QuestionsBatched       string
	EntitiesBatched        string
	StatementsBatched      string

	// LLMChunk asks the model to propose semantic cut points for one whole
	// document, returning verbatim boundary markers only; byte ranges are
	// recovered by aligning markers against the source, which keeps the
	// exact-slice chunk invariant out of the model's hands.
	LLMChunk string
}
