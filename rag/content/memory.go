package content

import (
	"context"
	"strings"
	"sync"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

const defaultMemoryMaxBatch = 256

// Memory is a bounded Store for in-memory experiments and deterministic tests.
// Production serving bundles use the SQLite implementation; Memory exists for
// callers that already own an explicit corpus slice and still need to exercise
// the same ID-bounded lookup contract.
type Memory struct {
	mu        sync.RWMutex
	documents map[string]rag.Document
	chunks    map[string]rag.Chunk
	maxBatch  int
	closed    bool
}

var _ Store = (*Memory)(nil)

// NewMemory copies the supplied corpus into an ID-addressed store. Documents
// may be omitted when a caller only needs chunk hydration; document and
// candidate-metadata lookups then fail closed.
func NewMemory(documents []rag.Document, chunks []rag.Chunk) (*Memory, error) {
	store := &Memory{
		documents: make(map[string]rag.Document, len(documents)),
		chunks:    make(map[string]rag.Chunk, len(chunks)),
		maxBatch:  defaultMemoryMaxBatch,
	}
	for _, document := range documents {
		if strings.TrimSpace(document.ID) == "" {
			return nil, errors.New("memory content document ID is required")
		}
		if _, duplicate := store.documents[document.ID]; duplicate {
			return nil, errors.Errorf("memory content contains duplicate document %q", document.ID)
		}
		store.documents[document.ID] = document
	}
	for _, chunk := range chunks {
		if strings.TrimSpace(chunk.ID) == "" || strings.TrimSpace(chunk.DocumentID) == "" {
			return nil, errors.New("memory content chunk identity is incomplete")
		}
		if _, duplicate := store.chunks[chunk.ID]; duplicate {
			return nil, errors.Errorf("memory content contains duplicate chunk %q", chunk.ID)
		}
		store.chunks[chunk.ID] = chunk
	}
	return store, nil
}

func (m *Memory) Documents(ctx context.Context, ids []string) ([]rag.Document, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if err := m.validateIDs(ctx, ids); err != nil {
		return nil, err
	}
	result := make([]rag.Document, 0, len(ids))
	for _, id := range ids {
		document, ok := m.documents[id]
		if !ok {
			return nil, errors.Errorf("load memory content document %q: not found", id)
		}
		result = append(result, document)
	}
	return result, nil
}

func (m *Memory) Chunks(ctx context.Context, ids []string) ([]rag.Chunk, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if err := m.validateIDs(ctx, ids); err != nil {
		return nil, err
	}
	result := make([]rag.Chunk, 0, len(ids))
	for _, id := range ids {
		chunk, ok := m.chunks[id]
		if !ok {
			return nil, errors.Errorf("load memory content chunk %q: not found", id)
		}
		result = append(result, chunk)
	}
	return result, nil
}

func (m *Memory) CandidateMetadata(ctx context.Context, ids []string) ([]CandidateMetadata, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if err := m.validateIDs(ctx, ids); err != nil {
		return nil, err
	}
	result := make([]CandidateMetadata, 0, len(ids))
	for _, id := range ids {
		chunk, ok := m.chunks[id]
		if !ok {
			return nil, errors.Errorf("load memory content chunk %q: not found", id)
		}
		document, ok := m.documents[chunk.DocumentID]
		if !ok {
			return nil, errors.Errorf("load memory content document %q: not found", chunk.DocumentID)
		}
		result = append(result, CandidateMetadata{
			ChunkID: chunk.ID, DocumentID: chunk.DocumentID,
			Metadata: cloneMetadata(document.Metadata),
		})
	}
	return result, nil
}

func (m *Memory) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

func (m *Memory) validateIDs(ctx context.Context, ids []string) error {
	if m == nil || m.closed {
		return errors.New("memory content store is not open")
	}
	if ctx == nil {
		return errors.New("content lookup context is required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(ids) == 0 {
		return errors.New("content lookup requires at least one ID")
	}
	if len(ids) > m.maxBatch {
		return errors.Errorf("content lookup batch size %d exceeds maximum %d", len(ids), m.maxBatch)
	}
	seen := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		if strings.TrimSpace(id) == "" {
			return errors.New("content lookup ID cannot be empty")
		}
		if _, duplicate := seen[id]; duplicate {
			return errors.Errorf("content lookup contains duplicate ID %q", id)
		}
		seen[id] = struct{}{}
	}
	return nil
}

func cloneMetadata(source map[string]string) map[string]string {
	if source == nil {
		return nil
	}
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}
