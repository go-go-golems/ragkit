package indexbundle

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
)

// TestLexicalOnlyBundleBuildsAndOpens covers the lexical-only serving path:
// a bundle built without an embedding identity must open without a query
// embedder and search through its lexical index.
func TestLexicalOnlyBundleBuildsAndOpens(t *testing.T) {
	ctx := context.Background()
	documents := []rag.Document{
		{
			ID: "doc-1", SourceURI: "fixture://doc-1", Title: "Gold Eagle Guide",
			Text: "# Overview\nThe American Gold Eagle is a bullion coin.\n\n# Grading\nProof coins carry mirror finishes.",
		},
	}
	documents[0].ContentDigest = digest.Text(documents[0].Text)
	chunks := []rag.Chunk{
		{
			ID: "chunk-1", DocumentID: "doc-1", Ordinal: 0,
			Range: rag.Range{ByteStart: 0, ByteEnd: len(documents[0].Text)},
			Text:  documents[0].Text, Chunker: "fixture-v1",
		},
	}
	chunks[0].ContentDigest = digest.Text(chunks[0].Text)
	representations, err := rag.RawRepresentations(chunks)
	require.NoError(t, err)

	result, err := Build(ctx, BuildInput{
		OutputRoot:      t.TempDir(),
		CorpusPath:      "corpus.json",
		Documents:       documents,
		Chunks:          chunks,
		Representations: representations,
		Chunker:         ChunkerIdentity{Name: "fixture-v1", MaximumRunes: 4000},
	})
	require.NoError(t, err)
	require.False(t, result.Reused)

	bundle, err := Open(ctx, OpenOptions{Path: result.Path, ScratchDirectory: t.TempDir()})
	require.NoError(t, err)
	defer func() { require.NoError(t, bundle.Close()) }()
	require.Nil(t, bundle.Vector)
	require.NotNil(t, bundle.Lexical)

	hits, err := bundle.Lexical.Search(ctx, rag.Query{ID: "q", Text: "gold eagle proof"}, 5)
	require.NoError(t, err)
	require.NotEmpty(t, hits)
	require.Equal(t, "chunk-1", hits[0].ChunkID)
}
