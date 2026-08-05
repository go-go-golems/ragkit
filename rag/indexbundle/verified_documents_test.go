package indexbundle

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func TestLoadVerifiedDocumentsLoadsManifestCorpusUnderRoot(t *testing.T) {
	root := t.TempDir()
	documents := testCorpus()
	data, err := json.Marshal(documents)
	require.NoError(t, err)
	corpusPath := filepath.Join(root, "datasets", "ttc", "corpus.json")
	require.NoError(t, os.MkdirAll(filepath.Dir(corpusPath), 0o755))
	require.NoError(t, os.WriteFile(corpusPath, data, 0o644))
	corpusDigest, err := digest.JSON(documents)
	require.NoError(t, err)
	bundlePath := writeBundle(t, Manifest{
		CorpusPath: filepath.Join("datasets", "ttc", "corpus.json"), CorpusDigest: corpusDigest, DocumentCount: len(documents),
	}, testChunks())

	loaded, err := LoadVerifiedDocuments(t.Context(), bundlePath, root)
	require.NoError(t, err)
	require.Equal(t, documents, loaded)
}

func TestLoadVerifiedDocumentsRejectsUntrustedCorpusIdentity(t *testing.T) {
	t.Run("digest mismatch", func(t *testing.T) {
		root := t.TempDir()
		corpusPath, _ := writeCorpus(t, testCorpus())
		inside := filepath.Join(root, "corpus.json")
		data, err := os.ReadFile(corpusPath)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(inside, data, 0o644))
		bundlePath := writeBundle(t, Manifest{CorpusPath: "corpus.json", CorpusDigest: "wrong", DocumentCount: 2}, testChunks())
		_, err = LoadVerifiedDocuments(t.Context(), bundlePath, root)
		require.ErrorContains(t, err, "digest")
	})

	t.Run("duplicate document", func(t *testing.T) {
		root := t.TempDir()
		documents := []rag.Document{{ID: "duplicate", Text: "first"}, {ID: "duplicate", Text: "second"}}
		data, err := json.Marshal(documents)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(root, "corpus.json"), data, 0o644))
		corpusDigest, err := digest.JSON(documents)
		require.NoError(t, err)
		bundlePath := writeBundle(t, Manifest{CorpusPath: "corpus.json", CorpusDigest: corpusDigest, DocumentCount: 2}, testChunks())
		_, err = LoadVerifiedDocuments(t.Context(), bundlePath, root)
		require.ErrorContains(t, err, "duplicate document")
	})

	t.Run("document count mismatch", func(t *testing.T) {
		root := t.TempDir()
		documents := testCorpus()
		data, err := json.Marshal(documents)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(root, "corpus.json"), data, 0o644))
		corpusDigest, err := digest.JSON(documents)
		require.NoError(t, err)
		bundlePath := writeBundle(t, Manifest{CorpusPath: "corpus.json", CorpusDigest: corpusDigest, DocumentCount: 99}, testChunks())
		_, err = LoadVerifiedDocuments(t.Context(), bundlePath, root)
		require.ErrorContains(t, err, "99")
	})
}

func TestLoadVerifiedDocumentsRejectsPathsOutsideRoot(t *testing.T) {
	root := t.TempDir()
	outsidePath, corpusDigest := writeCorpus(t, testCorpus())

	for name, candidate := range map[string]string{
		"relative escape": filepath.Join("..", filepath.Base(filepath.Dir(outsidePath)), filepath.Base(outsidePath)),
		"absolute escape": outsidePath,
	} {
		t.Run(name, func(t *testing.T) {
			bundlePath := writeBundle(t, Manifest{CorpusPath: candidate, CorpusDigest: corpusDigest, DocumentCount: 2}, testChunks())
			_, err := LoadVerifiedDocuments(t.Context(), bundlePath, root)
			require.ErrorContains(t, err, "escapes")
		})
	}
}
