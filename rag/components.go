package rag

import (
	"context"
	"io"
)

// Chunker splits one document while preserving source lineage.
type Chunker interface {
	Name() string
	Chunk(context.Context, Document) ([]Chunk, error)
}

// GenerationRequest describes one typed text-generation operation.
type GenerationRequest struct {
	Kind         string     `json:"kind"`
	Model        string     `json:"model"`
	Prompt       string     `json:"prompt"`
	Text         string     `json:"text,omitempty"`
	Evidence     []Evidence `json:"evidence,omitempty"`
	OutputSchema string     `json:"output_schema,omitempty"`
}

// GenerationResult is provider text plus optional usage.
type GenerationResult struct {
	Text         string   `json:"text"`
	Citations    []string `json:"citations,omitempty"`
	FinishReason string   `json:"finish_reason,omitempty"`
	Usage        Usage    `json:"usage"`
}

// Generator performs one text-generation operation.
type Generator interface {
	Generate(context.Context, GenerationRequest) (GenerationResult, error)
}

// EmbeddingRequest is one provider batch.
type EmbeddingRequest struct {
	Model string   `json:"model"`
	Texts []string `json:"texts"`
}

// EmbeddingResult contains one vector per input text.
type EmbeddingResult struct {
	Vectors [][]float32 `json:"vectors"`
	Usage   Usage       `json:"usage"`
}

// Embedder embeds a batch of text.
type Embedder interface {
	Embed(context.Context, EmbeddingRequest) (EmbeddingResult, error)
}

// Searcher searches one already-built lexical or vector index.
type Searcher interface {
	Search(context.Context, Query, int) ([]Hit, error)
}

// Index combines search with resource cleanup.
type Index interface {
	Searcher
	io.Closer
}

// RerankRequest describes one reranking call.
type RerankRequest struct {
	Model      string     `json:"model"`
	Query      Query      `json:"query"`
	Candidates []Evidence `json:"candidates"`
	Results    int        `json:"results"`
}

// RerankResult returns source evidence in final rank order.
type RerankResult struct {
	Evidence []Evidence `json:"evidence"`
	Usage    Usage      `json:"usage"`
}

// Reranker reorders hydrated source evidence.
type Reranker interface {
	Rerank(context.Context, RerankRequest) (RerankResult, error)
}
