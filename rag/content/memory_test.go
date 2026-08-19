package content

import (
	"context"
	"fmt"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func TestMemoryPreservesCallerOrderAndFailsClosed(t *testing.T) {
	documents := []rag.Document{
		{ID: "doc-a", Metadata: map[string]string{"scope": "public"}},
		{ID: "doc-b", Metadata: map[string]string{"scope": "staff"}},
	}
	chunks := []rag.Chunk{
		{ID: "chunk-a", DocumentID: "doc-a", Text: "alpha"},
		{ID: "chunk-b", DocumentID: "doc-b", Text: "bravo"},
	}
	store, err := NewMemory(documents, chunks)
	require.NoError(t, err)
	documents[0].Metadata["scope"] = "mutated-input"

	loaded, err := store.Chunks(t.Context(), []string{"chunk-b", "chunk-a"})
	require.NoError(t, err)
	require.Equal(t, []string{"chunk-b", "chunk-a"}, []string{loaded[0].ID, loaded[1].ID})

	loadedDocuments, err := store.Documents(t.Context(), []string{"doc-a"})
	require.NoError(t, err)
	require.Equal(t, "public", loadedDocuments[0].Metadata["scope"])
	loadedDocuments[0].Metadata["scope"] = "mutated-result"

	metadata, err := store.CandidateMetadata(t.Context(), []string{"chunk-a"})
	require.NoError(t, err)
	require.Equal(t, "public", metadata[0].Metadata["scope"])
	metadata[0].Metadata["scope"] = "mutated-metadata"
	metadata, err = store.CandidateMetadata(t.Context(), []string{"chunk-a"})
	require.NoError(t, err)
	require.Equal(t, "public", metadata[0].Metadata["scope"])

	_, err = store.Chunks(t.Context(), []string{"missing"})
	require.ErrorContains(t, err, "not found")
	_, err = store.Chunks(t.Context(), []string{"chunk-a", "chunk-a"})
	require.ErrorContains(t, err, "duplicate")

	require.NoError(t, store.Close())
	_, err = store.Documents(context.Background(), []string{"doc-a"})
	require.ErrorContains(t, err, "not open")
}

func TestLoadChunksSplitsAtStoreBatchBoundary(t *testing.T) {
	chunks := make([]rag.Chunk, DefaultMaxBatch+1)
	ids := make([]string, len(chunks))
	for index := range chunks {
		id := fmt.Sprintf("chunk-%03d", index)
		chunks[index] = rag.Chunk{ID: id, DocumentID: "doc", Text: id}
		ids[index] = id
	}
	store, err := NewMemory(nil, chunks)
	require.NoError(t, err)
	loaded, err := LoadChunks(t.Context(), store, ids)
	require.NoError(t, err)
	require.Len(t, loaded, len(chunks))
	require.Equal(t, ids[0], loaded[0].ID)
	require.Equal(t, ids[len(ids)-1], loaded[len(loaded)-1].ID)
}

func TestMemoryRejectsDuplicateCorpusIdentities(t *testing.T) {
	_, err := NewMemory([]rag.Document{{ID: "doc"}, {ID: "doc"}}, nil)
	require.ErrorContains(t, err, "duplicate document")
	_, err = NewMemory(nil, []rag.Chunk{{ID: "chunk", DocumentID: "doc"}, {ID: "chunk", DocumentID: "doc"}})
	require.ErrorContains(t, err, "duplicate chunk")
}
