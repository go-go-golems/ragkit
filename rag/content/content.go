// Package content defines bounded source-content lookup used by serving
// retrieval indexes. Search backends return identities; a content Store
// resolves only the bounded candidate set that a caller explicitly requests.
package content

import (
	"context"

	"github.com/go-go-golems/ragkit/rag"
)

// CandidateMetadata is the authorization-only projection of one chunk. It
// intentionally excludes source text so authorization cannot accidentally
// hydrate evidence before access checks are complete.
type CandidateMetadata struct {
	ChunkID    string
	DocumentID string
	Metadata   map[string]string
}

// Store provides bounded source lookups for a serving bundle. Implementations
// must reject duplicate or oversized ID batches and must fail closed on a
// missing ID. There is deliberately no method that loads the complete corpus.
type Store interface {
	Documents(context.Context, []string) ([]rag.Document, error)
	Chunks(context.Context, []string) ([]rag.Chunk, error)
	CandidateMetadata(context.Context, []string) ([]CandidateMetadata, error)
	Close() error
}
