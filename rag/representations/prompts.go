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
// Callers targeting a specific domain should construct their own PromptSet;
// DefaultPromptSet returns the original rag-ttc plant-care prompts verbatim,
// which serve as working examples and keep cache compatibility for TTC
// corpora.
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

// DefaultPromptSet returns the canonical rag-ttc prompt texts, verbatim.
// They are domain-flavored (plant care); production corpora in other domains
// should define their own set with the same shape.
func DefaultPromptSet() PromptSet {
	return PromptSet{
		AdapterVersion:         "ttc-representation-adapter-v1",
		ContextPolicy:          "chunk-text-v1",
		Contextual:             "You see one chunk of a plant-care document together with the document title, its section path, and the document lead. Write two or three sentences (at most 80 words) that situate the chunk within the document for search retrieval: name the species or product concerned and say what the chunk covers. Output only those sentences.",
		Summary2Sent:           "Summarize the following plant-care text in two sentences, keeping species names and concrete numbers.",
		SummaryKeywords:        "Compress the following plant-care text into one telegraphic line: species names, common names, main topics, and every concrete number. Output only the line, no sentences.",
		Questions:              "Write three short questions that a plant buyer might type into a search box and that the following text answers. Keep species and product names. Output one question per line with no numbering.",
		Entities:               "List the plant species, common names, botanical synonyms, and product names the following text concerns. Output one comma-separated line and nothing else.",
		Statements:             "List the atomic factual statements the following plant-care text asserts. Split compound sentences into separate statements. Do not add information and do not evaluate it. Output one statement per line with no numbering.",
		DocumentSummary:        "You see one full plant-care document. Write a summary of three to five sentences for search retrieval indexing: name the species or product the document concerns, say what topics it covers, and keep the most important concrete numbers. Output only the summary.",
		ContextualBatched:      `You see one plant-care document's title, its opening, and a numbered list of chunks from it, each with its section path. For EACH numbered chunk, write two or three sentences (at most 80 words) that situate that chunk within the document for search retrieval: name the species or product concerned and say what the chunk covers. Respond with a JSON array only, one object per chunk: [{"chunk": <number>, "text": "<the sentences>"}]. Include every chunk number exactly once.`,
		SummaryKeywordsBatched: `You see a numbered list of chunks from one plant-care document. For EACH numbered chunk, compress it into one telegraphic line: species names, common names, main topics, and every concrete number. Respond with a JSON array only, one object per chunk: [{"chunk": <number>, "text": "<the line>"}]. Include every chunk number exactly once.`,
		QuestionsBatched:       `You see a numbered list of chunks from one plant-care document. For EACH numbered chunk, write three short questions that a plant buyer might type into a search box and that the chunk answers. Keep species and product names. Respond with a JSON array only, one object per chunk: [{"chunk": <number>, "text": "<the three questions, separated by newlines>"}]. Include every chunk number exactly once.`,
		EntitiesBatched:        `You see a numbered list of chunks from one plant-care document. For EACH numbered chunk, list the plant species, common names, botanical synonyms, and product names it concerns as one comma-separated line. Respond with a JSON array only, one object per chunk: [{"chunk": <number>, "text": "<the line>"}]. Include every chunk number exactly once.`,
		StatementsBatched:      `You see a numbered list of chunks from one plant-care document. For EACH numbered chunk, list the atomic factual statements it asserts, one per line. Split compound sentences into separate statements. Do not add information and do not evaluate it. Respond with a JSON array only, one object per chunk: [{"chunk": <number>, "text": "<the statements, one per line, separated by newlines>"}]. Include every chunk number exactly once.`,
		LLMChunk:               `You see one full plant-care document. Split it into retrieval chunks of roughly 500 to 2500 characters. Each chunk must cover one coherent topic; cut where the topic changes. Respond with a JSON array only, one object per chunk in document order: [{"start": "<the exact first 8 to 12 words of the chunk, copied verbatim from the document>"}]. The first chunk starts at the very beginning of the document.`,
	}
}
