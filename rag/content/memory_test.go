package content

import (
	"context"
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

	loaded, err := store.Chunks(t.Context(), []string{"chunk-b", "chunk-a"})
	require.NoError(t, err)
	require.Equal(t, []string{"chunk-b", "chunk-a"}, []string{loaded[0].ID, loaded[1].ID})

	metadata, err := store.CandidateMetadata(t.Context(), []string{"chunk-a"})
	require.NoError(t, err)
	require.Equal(t, "public", metadata[0].Metadata["scope"])
	metadata[0].Metadata["scope"] = "mutated"
	require.Equal(t, "public", documents[0].Metadata["scope"])

	_, err = store.Chunks(t.Context(), []string{"missing"})
	require.ErrorContains(t, err, "not found")
	_, err = store.Chunks(t.Context(), []string{"chunk-a", "chunk-a"})
	require.ErrorContains(t, err, "duplicate")

	require.NoError(t, store.Close())
	_, err = store.Documents(context.Background(), []string{"doc-a"})
	require.ErrorContains(t, err, "not open")
}

func TestMemoryRejectsDuplicateCorpusIdentities(t *testing.T) {
	_, err := NewMemory([]rag.Document{{ID: "doc"}, {ID: "doc"}}, nil)
	require.ErrorContains(t, err, "duplicate document")
	_, err = NewMemory(nil, []rag.Chunk{{ID: "chunk", DocumentID: "doc"}, {ID: "chunk", DocumentID: "doc"}})
	require.ErrorContains(t, err, "duplicate chunk")
}
