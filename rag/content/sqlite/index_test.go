package sqlite

import (
	"context"
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func TestBuildOpenAndBoundedLookups(t *testing.T) {
	ctx := context.Background()
	documents := []rag.Document{
		{ID: "doc-1", SourceURI: "db://1", Title: "One", Text: "alpha beta", ContentDigest: digest.Text("alpha beta"), Metadata: map[string]string{"scope": "public", "owner": "buying"}},
		{ID: "doc-2", SourceURI: "db://2", Title: "Two", Text: "gamma delta", ContentDigest: digest.Text("gamma delta"), Metadata: map[string]string{"scope": "internal", "owner": "logistics"}},
	}
	chunks := []rag.Chunk{
		{ID: "chunk-1", DocumentID: "doc-1", Ordinal: 0, Range: rag.Range{ByteStart: 0, ByteEnd: 5}, Text: "alpha", ContentDigest: digest.Text("alpha"), Chunker: "fixed-v1"},
		{ID: "chunk-2", DocumentID: "doc-2", Ordinal: 0, Range: rag.Range{ByteStart: 0, ByteEnd: 5}, Text: "gamma", ContentDigest: digest.Text("gamma"), Chunker: "fixed-v1"},
	}
	path := t.TempDir() + "/content.sqlite"
	result, err := Build(ctx, BuildInput{
		Path: path,
		Documents: func(ctx context.Context, yield func(rag.Document) error) error {
			for _, document := range documents {
				if err := yield(document); err != nil {
					return err
				}
			}
			return nil
		},
		Chunks: func(ctx context.Context, yield func(rag.Chunk) error) error {
			for _, chunk := range chunks {
				if err := yield(chunk); err != nil {
					return err
				}
			}
			return nil
		},
	})
	require.NoError(t, err)
	require.Equal(t, path, result.Path)
	require.Equal(t, 2, result.Identity.DocumentCount)
	require.Equal(t, 2, result.Identity.ChunkCount)
	require.Equal(t, Backend, result.Identity.Backend)

	index, identity, err := Open(ctx, Config{Path: path, MaxBatch: 2})
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, index.Close()) })
	require.Equal(t, result.Identity, identity)

	gotDocuments, err := index.Documents(ctx, []string{"doc-2", "doc-1"})
	require.NoError(t, err)
	require.Equal(t, []string{"doc-2", "doc-1"}, []string{gotDocuments[0].ID, gotDocuments[1].ID})
	require.Equal(t, "internal", gotDocuments[0].Metadata["scope"])

	gotChunks, err := index.Chunks(ctx, []string{"chunk-2", "chunk-1"})
	require.NoError(t, err)
	require.Equal(t, []string{"chunk-2", "chunk-1"}, []string{gotChunks[0].ID, gotChunks[1].ID})
	require.Equal(t, "gamma", gotChunks[0].Text)

	metadata, err := index.CandidateMetadata(ctx, []string{"chunk-2", "chunk-1"})
	require.NoError(t, err)
	require.Equal(t, []string{"chunk-2", "chunk-1"}, []string{metadata[0].ChunkID, metadata[1].ChunkID})
	require.Equal(t, "doc-2", metadata[0].DocumentID)
	require.Equal(t, "logistics", metadata[0].Metadata["owner"])
}

func TestLookupBoundsAndMissingIDsFailClosed(t *testing.T) {
	ctx := context.Background()
	document := rag.Document{ID: "doc-1", Text: "alpha", ContentDigest: digest.Text("alpha")}
	chunk := rag.Chunk{ID: "chunk-1", DocumentID: "doc-1", Ordinal: 0, Range: rag.Range{ByteStart: 0, ByteEnd: 5}, Text: "alpha", ContentDigest: digest.Text("alpha"), Chunker: "fixed-v1"}
	path := t.TempDir() + "/content.sqlite"
	_, err := Build(ctx, BuildInput{
		Path:      path,
		Documents: func(context.Context, func(rag.Document) error) error { return nil },
		Chunks:    func(context.Context, func(rag.Chunk) error) error { return nil },
	})
	// Empty producers are rejected before a serving index can be published.
	require.Error(t, err)

	// Build a valid one-row fixture for lookup validation.
	_, err = Build(ctx, BuildInput{
		Path:      path,
		Documents: func(_ context.Context, yield func(rag.Document) error) error { return yield(document) },
		Chunks:    func(_ context.Context, yield func(rag.Chunk) error) error { return yield(chunk) },
	})
	require.NoError(t, err)
	index, _, err := Open(ctx, Config{Path: path, MaxBatch: 1})
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, index.Close()) })

	_, err = index.Chunks(ctx, []string{"chunk-1", "chunk-1"})
	require.Error(t, err)
	_, err = index.Chunks(ctx, []string{"chunk-1", "chunk-2"})
	require.Error(t, err)
	_, err = index.Chunks(ctx, []string{"chunk-1", "chunk-2", "chunk-3"})
	require.Error(t, err)
	_, err = index.CandidateMetadata(ctx, []string{"missing"})
	require.Error(t, err)
}
