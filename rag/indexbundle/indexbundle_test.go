package indexbundle

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/go-go-golems/ragkit/rag/chunking"
	"github.com/go-go-golems/ragkit/rag/embedding"
	"github.com/stretchr/testify/require"
)

func fixtureInput(t *testing.T, outputRoot string) BuildInput {
	t.Helper()
	documents := []rag.Document{{
		ID: "doc-1", Title: "Trees",
		Text:          "# Oak\nOak trees have lobed leaves.",
		ContentDigest: "doc-digest",
	}}
	chunker := &chunking.Markdown{MaxSectionRunes: 1200, OverlapRunes: 120}
	chunks, err := chunking.Apply(t.Context(), chunker, documents)
	require.NoError(t, err)
	representations, err := rag.RawRepresentations(chunks)
	require.NoError(t, err)
	hash := &embedding.HashEmbedder{Dimensions: 16}
	texts := make([]string, len(representations))
	for i := range representations {
		texts[i] = representations[i].Text
	}
	embedded, err := hash.Embed(t.Context(), rag.EmbeddingRequest{
		Model: "hash-v1-d16", Texts: texts,
	})
	require.NoError(t, err)
	vectors := make([]rag.Vector, len(representations))
	for i := range representations {
		vectors[i] = rag.Vector{
			RepresentationID: representations[i].ID,
			Values:           embedded.Vectors[i], Model: "hash-v1-d16",
		}
	}
	return BuildInput{
		OutputRoot: outputRoot, CorpusPath: "fixture/corpus.json",
		Documents: documents, Chunks: chunks, Representations: representations,
		Vectors: vectors, QueryEmbedder: hash,
		Chunker: ChunkerIdentity{
			Name: chunker.Name(), MaximumRunes: 1200, OverlapRunes: 120,
		},
		Embedding: &VectorIdentity{
			Backend: "sqlite-exact", Version: 1, Channel: "vector",
			Provider: "hash", Model: "hash-v1-d16", Dimensions: 16,
		},
	}
}

func TestBuildIsDeterministicReusableAndOpenable(t *testing.T) {
	input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
	first, err := Build(t.Context(), input)
	require.NoError(t, err)
	require.False(t, first.Reused)
	require.DirExists(t, filepath.Join(first.Path, "bleve"))
	require.FileExists(t, filepath.Join(first.Path, "vectors.sqlite"))
	require.FileExists(t, filepath.Join(first.Path, "chunks.json"))
	require.Positive(t, first.BleveBytes)
	require.Positive(t, first.VectorBytes)

	second, err := Build(t.Context(), input)
	require.NoError(t, err)
	require.True(t, second.Reused)
	require.Equal(t, first.Manifest.BundleID, second.Manifest.BundleID)
	require.Equal(t, first.Path, second.Path)

	bundle, err := Open(t.Context(), OpenOptions{
		Path: first.Path, QueryEmbedder: input.QueryEmbedder,
		EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 16,
	})
	require.NoError(t, err)
	require.Equal(t, input.Chunks, bundle.Chunks)
	lexical, err := bundle.Lexical.Search(
		t.Context(), rag.Query{ID: "q", Text: "lobed leaves"}, 5,
	)
	require.NoError(t, err)
	require.NotEmpty(t, lexical)
	vector, err := bundle.Vector.Search(
		t.Context(), rag.Query{ID: "q", Text: "lobed leaves"}, 5,
	)
	require.NoError(t, err)
	require.NotEmpty(t, vector)
	require.NoError(t, bundle.Close())
	require.NoError(t, bundle.Close(), "close must be idempotent")
}

func TestOpenAllowsAdmittedDocumentWithoutChunks(t *testing.T) {
	input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
	input.Documents = append(input.Documents, rag.Document{
		ID: "doc-package-only", Title: "Package only", Text: "package trees",
		ContentDigest: "package-only-digest",
	})
	result, err := Build(t.Context(), input)
	require.NoError(t, err)
	require.Equal(t, 2, result.Manifest.DocumentCount)

	bundle, err := Open(t.Context(), OpenOptions{
		Path: result.Path, QueryEmbedder: input.QueryEmbedder,
		EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 16,
	})
	require.NoError(t, err)
	require.NoError(t, bundle.Close())
}

func TestIdentityChangesForEverySemanticInput(t *testing.T) {
	base := fixtureInput(t, t.TempDir())
	id, _, _, _, err := CalculateID(
		base.Documents, base.Representations, base.Chunker,
		BackendIdentity{Backend: "bleve-bm25", Version: 1, Channel: "bm25", TitleBoost: 2, BodyBoost: 1},
		base.Embedding,
	)
	require.NoError(t, err)

	changedChunker := base.Chunker
	changedChunker.MaximumRunes++
	other, _, _, _, err := CalculateID(
		base.Documents, base.Representations, changedChunker,
		BackendIdentity{Backend: "bleve-bm25", Version: 1, Channel: "bm25", TitleBoost: 2, BodyBoost: 1},
		base.Embedding,
	)
	require.NoError(t, err)
	require.NotEqual(t, id, other)

	changedEmbedding := *base.Embedding
	changedEmbedding.Model = "different"
	other, _, _, _, err = CalculateID(
		base.Documents, base.Representations, base.Chunker,
		BackendIdentity{Backend: "bleve-bm25", Version: 1, Channel: "bm25", TitleBoost: 2, BodyBoost: 1},
		&changedEmbedding,
	)
	require.NoError(t, err)
	require.NotEqual(t, id, other)

	changedRepresentations := append([]rag.Representation(nil), base.Representations...)
	changedRepresentations[0].Text += " changed"
	other, _, _, _, err = CalculateID(
		base.Documents, changedRepresentations, base.Chunker,
		BackendIdentity{Backend: "bleve-bm25", Version: 1, Channel: "bm25", TitleBoost: 2, BodyBoost: 1},
		base.Embedding,
	)
	require.NoError(t, err)
	require.NotEqual(t, id, other)
}

func TestBuildCleansPartialDirectoryOnFailure(t *testing.T) {
	root := filepath.Join(t.TempDir(), "indexes")
	input := fixtureInput(t, root)
	input.Representations[0].ChunkID = "missing"
	_, err := Build(t.Context(), input)
	require.Error(t, err)
	matches, globErr := filepath.Glob(filepath.Join(root, ".bundle-partial-*"))
	require.NoError(t, globErr)
	require.Empty(t, matches)
}

func TestBuildRejectsIncompleteExistingBundle(t *testing.T) {
	input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
	result, err := Build(t.Context(), input)
	require.NoError(t, err)
	require.NoError(t, os.Remove(filepath.Join(result.Path, "chunks.json")))
	_, err = Build(t.Context(), input)
	require.ErrorContains(t, err, "existing bundle is incomplete")
}

func TestOpenRejectsEmbeddingAndManifestMismatches(t *testing.T) {
	input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
	result, err := Build(t.Context(), input)
	require.NoError(t, err)

	_, err = Open(t.Context(), OpenOptions{
		Path: result.Path, QueryEmbedder: input.QueryEmbedder,
		EmbeddingModel: "other", EmbeddingDimensions: 16,
	})
	require.ErrorContains(t, err, "differs from bundle model")
	_, err = Open(t.Context(), OpenOptions{
		Path: result.Path, QueryEmbedder: input.QueryEmbedder,
		EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 8,
	})
	require.ErrorContains(t, err, "dimensions")

	manifestPath := filepath.Join(result.Path, "manifest.json")
	data, err := os.ReadFile(manifestPath)
	require.NoError(t, err)
	var manifest Manifest
	require.NoError(t, json.Unmarshal(data, &manifest))
	manifest.ChunkCount++
	data, err = json.Marshal(manifest)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(manifestPath, data, 0o600))
	_, err = Open(context.Background(), OpenOptions{
		Path: result.Path, QueryEmbedder: input.QueryEmbedder,
		EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 16,
	})
	require.ErrorContains(t, err, "counts differ")
}
