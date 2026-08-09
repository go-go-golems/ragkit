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

func verifiedDocumentsFixture(t *testing.T) (string, string, []rag.Document) {
	t.Helper()
	root := t.TempDir()
	input := fixtureInput(t, filepath.Join(root, "indexes"))
	input.CorpusPath = "corpus.json"
	input.Documents = append(input.Documents, rag.Document{
		ID: "doc-2", Title: "Maples", Text: "Maples have winged seeds.",
		ContentDigest: digest.Text("Maples have winged seeds."),
	})
	data, err := json.Marshal(input.Documents)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(root, input.CorpusPath), data, 0o600))
	result, err := Build(t.Context(), input)
	require.NoError(t, err)
	return root, result.Path, input.Documents
}

func rewriteManifest(t *testing.T, bundlePath string, mutate func(*Manifest)) {
	t.Helper()
	path := filepath.Join(bundlePath, manifestName)
	var manifest Manifest
	require.NoError(t, readJSON(path, &manifest))
	mutate(&manifest)
	data, err := json.Marshal(manifest)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(path, data, 0o600))
}

func TestLoadVerifiedDocumentsLoadsManifestCorpusUnderRoot(t *testing.T) {
	root, bundlePath, documents := verifiedDocumentsFixture(t)
	loaded, err := LoadVerifiedDocuments(t.Context(), bundlePath, root)
	require.NoError(t, err)
	require.Equal(t, documents, loaded)
}

func TestLoadVerifiedDocumentsRejectsTamperedManifestIdentity(t *testing.T) {
	root, bundlePath, _ := verifiedDocumentsFixture(t)
	replacement := []rag.Document{{ID: "replacement", Text: "other corpus"}}
	data, err := json.Marshal(replacement)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(root, "replacement.json"), data, 0o600))
	replacementDigest, err := digest.JSON(replacement)
	require.NoError(t, err)
	rewriteManifest(t, bundlePath, func(manifest *Manifest) {
		manifest.CorpusPath = "replacement.json"
		manifest.CorpusDigest = replacementDigest
		manifest.DocumentCount = len(replacement)
	})

	_, err = LoadVerifiedDocuments(t.Context(), bundlePath, root)
	require.ErrorContains(t, err, "bundle identity")
}

func TestLoadVerifiedDocumentsRejectsUntrustedCorpusIdentity(t *testing.T) {
	t.Run("digest mismatch", func(t *testing.T) {
		root, bundlePath, documents := verifiedDocumentsFixture(t)
		documents[0].Title = "tampered"
		data, err := json.Marshal(documents)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(root, "corpus.json"), data, 0o600))
		_, err = LoadVerifiedDocuments(t.Context(), bundlePath, root)
		require.ErrorContains(t, err, "digest")
	})

	t.Run("duplicate document", func(t *testing.T) {
		root, bundlePath, documents := verifiedDocumentsFixture(t)
		documents[1].ID = documents[0].ID
		data, err := json.Marshal(documents)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(root, "corpus.json"), data, 0o600))
		_, err = LoadVerifiedDocuments(t.Context(), bundlePath, root)
		require.ErrorContains(t, err, "duplicate document")
	})

	t.Run("document count mismatch", func(t *testing.T) {
		root, bundlePath, documents := verifiedDocumentsFixture(t)
		data, err := json.Marshal(documents[:1])
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(root, "corpus.json"), data, 0o600))
		_, err = LoadVerifiedDocuments(t.Context(), bundlePath, root)
		require.ErrorContains(t, err, "2")
	})
}

func TestLoadVerifiedDocumentsRejectsPathsOutsideRoot(t *testing.T) {
	root, bundlePath, documents := verifiedDocumentsFixture(t)
	outside := filepath.Join(t.TempDir(), "corpus.json")
	data, err := json.Marshal(documents)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(outside, data, 0o600))
	rewriteManifest(t, bundlePath, func(manifest *Manifest) { manifest.CorpusPath = outside })

	_, err = LoadVerifiedDocuments(t.Context(), bundlePath, root)
	require.ErrorContains(t, err, "escapes")
}
