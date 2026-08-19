// Package content defines bounded source-content lookup used by serving
// retrieval indexes. Search backends return identities; a content Store
// resolves only the bounded candidate set that a caller explicitly requests.
package content

import (
	"context"
	"fmt"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
)

// DefaultMaxBatch is the fallback lookup batch size for stores that do not
// advertise a smaller implementation-specific limit.
const DefaultMaxBatch = 256

// BatchSizer is an optional Store capability used by bounded lookup helpers.
// Implementations should report the same positive limit enforced by their raw
// lookup methods.
type BatchSizer interface {
	MaxBatchSize() int
}

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

// LoadChunks resolves an arbitrarily sized, duplicate-free ID set through
// bounded Store calls. It preserves caller order and returns no partial result
// if any batch fails. Stores with a non-default limit should implement
// BatchSizer.
func LoadChunks(ctx context.Context, store Store, ids []string) ([]rag.Chunk, error) {
	if store == nil {
		return nil, fmt.Errorf("content store is required")
	}
	if len(ids) == 0 {
		return []rag.Chunk{}, nil
	}
	seen := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		if strings.TrimSpace(id) == "" {
			return nil, fmt.Errorf("content lookup ID cannot be empty")
		}
		if _, duplicate := seen[id]; duplicate {
			return nil, fmt.Errorf("content lookup contains duplicate ID %q", id)
		}
		seen[id] = struct{}{}
	}
	batchSize := DefaultMaxBatch
	if sized, ok := store.(BatchSizer); ok {
		batchSize = sized.MaxBatchSize()
		if batchSize < 1 {
			return nil, fmt.Errorf("content store maximum batch must be positive")
		}
	}
	result := make([]rag.Chunk, 0, len(ids))
	for start := 0; start < len(ids); start += batchSize {
		end := min(start+batchSize, len(ids))
		batch, err := store.Chunks(ctx, ids[start:end])
		if err != nil {
			return nil, fmt.Errorf("load content chunk batch [%d:%d]: %w", start, end, err)
		}
		result = append(result, batch...)
	}
	return result, nil
}
