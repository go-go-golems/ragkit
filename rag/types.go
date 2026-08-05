package rag

// Document is one immutable source revision.
type Document struct {
	ID            string            `json:"id"`
	SourceURI     string            `json:"source_uri,omitempty"`
	Title         string            `json:"title,omitempty"`
	Text          string            `json:"text"`
	ContentDigest string            `json:"content_digest"`
	Metadata      map[string]string `json:"metadata,omitempty"`
}

// Range is a half-open byte range in a source document.
type Range struct {
	ByteStart int `json:"byte_start"`
	ByteEnd   int `json:"byte_end"`
}

// Chunk is an exact slice of a source document.
type Chunk struct {
	ID            string `json:"id"`
	DocumentID    string `json:"document_id"`
	Ordinal       int    `json:"ordinal"`
	Range         Range  `json:"range"`
	Text          string `json:"text"`
	ContentDigest string `json:"content_digest"`
	Chunker       string `json:"chunker"`
}

// Representation is searchable text derived from a chunk. Raw chunk text is
// represented with Kind "raw"; generated summaries and questions use other
// kinds. A representation is retrieval material, not source evidence.
type Representation struct {
	ID            string `json:"id"`
	ChunkID       string `json:"chunk_id"`
	Kind          string `json:"kind"`
	Text          string `json:"text"`
	ContentDigest string `json:"content_digest"`
	Model         string `json:"model,omitempty"`
	PromptDigest  string `json:"prompt_digest,omitempty"`
}

// Vector is an embedding of one representation.
type Vector struct {
	RepresentationID string    `json:"representation_id"`
	Values           []float32 `json:"values"`
	Model            string    `json:"model"`
}

// Query is one evaluation or interactive retrieval request.
type Query struct {
	ID       string `json:"id"`
	Text     string `json:"text"`
	Split    string `json:"split,omitempty"`
	Category string `json:"category,omitempty"`
}

// Judgment assigns a relevance grade to a document or chunk for one query.
type Judgment struct {
	QueryID  string  `json:"query_id"`
	Target   string  `json:"target"`
	TargetID string  `json:"target_id"`
	Grade    float64 `json:"grade"`
}

// EvaluationSet binds labeled queries to one immutable corpus.
type EvaluationSet struct {
	ID           string     `json:"id"`
	CorpusDigest string     `json:"corpus_digest"`
	Queries      []Query    `json:"queries"`
	Judgments    []Judgment `json:"judgments"`
}

// Hit is one channel-local search result.
type Hit struct {
	RepresentationID string  `json:"representation_id"`
	ChunkID          string  `json:"chunk_id"`
	DocumentID       string  `json:"document_id"`
	Channel          string  `json:"channel"`
	Rank             int     `json:"rank"`
	Score            float64 `json:"score"`
}

// Contribution explains one channel's contribution to a fused result.
type Contribution struct {
	Channel string  `json:"channel"`
	Rank    int     `json:"rank"`
	Weight  float64 `json:"weight"`
	Value   float64 `json:"value"`
}

// FusedHit combines channel-local results for one collapse target.
type FusedHit struct {
	ChunkID       string         `json:"chunk_id"`
	DocumentID    string         `json:"document_id"`
	Rank          int            `json:"rank"`
	Score         float64        `json:"score"`
	Contributions []Contribution `json:"contributions"`
}

// Evidence is hydrated source text suitable for reranking, generation, and
// human inspection.
type Evidence struct {
	Chunk          Chunk    `json:"chunk"`
	Rank           int      `json:"rank"`
	RetrievalScore float64  `json:"retrieval_score"`
	RerankerScore  *float64 `json:"reranker_score,omitempty"`
}

// Usage contains provider-reported usage. Pointers distinguish missing values
// from values explicitly reported as zero.
type Usage struct {
	InputTokens       *int64   `json:"input_tokens,omitempty"`
	CachedInputTokens *int64   `json:"cached_input_tokens,omitempty"`
	OutputTokens      *int64   `json:"output_tokens,omitempty"`
	ReasoningTokens   *int64   `json:"reasoning_tokens,omitempty"`
	EmbeddingTokens   *int64   `json:"embedding_tokens,omitempty"`
	CostUSD           *float64 `json:"cost_usd,omitempty"`
}
