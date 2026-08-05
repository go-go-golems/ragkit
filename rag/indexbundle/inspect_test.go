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

// realBundlePath is the bundle every measured number in the design doc comes
// from. The reader tests that assert exact counts are gated on it, because a
// developer without the cache must still be able to run the suite.
const realBundlePath = ".cache/rag-ttc/indexes/ttc-866972d249d18e770631e563346a4774"

// repoRoot walks up from the package directory to the module root, so a test
// can name a repository-relative path such as datasets/ttc/corpus.json. The
// manifest stores CorpusPath relative to the root, and Inspect resolves it
// against the working directory, which for a test is the package directory.
func repoRoot(t *testing.T) string {
	t.Helper()
	directory, err := os.Getwd()
	require.NoError(t, err)
	for range 10 {
		if _, err := os.Stat(filepath.Join(directory, "go.mod")); err == nil {
			return directory
		}
		parent := filepath.Dir(directory)
		require.NotEqual(t, parent, directory, "no go.mod above the package")
		directory = parent
	}
	t.Fatal("no go.mod within ten levels")
	return ""
}

// writeBundle writes the two files Inspect is allowed to read, and returns the
// directory.
func writeBundle(
	t *testing.T, manifest Manifest, chunks []rag.Chunk,
) string {
	t.Helper()
	directory := t.TempDir()
	manifest.SchemaVersion = SchemaVersion
	if manifest.BundleID == "" {
		manifest.BundleID = "ttc-test"
	}
	manifest.ChunkCount = len(chunks)
	write := func(name string, value any) {
		data, err := json.Marshal(value)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(filepath.Join(directory, name), data, 0o644))
	}
	write(manifestName, manifest)
	write(chunksName, chunks)
	return directory
}

func testChunks() []rag.Chunk {
	return []rag.Chunk{
		{
			ID: "chunk-a0", DocumentID: "doc-a", Ordinal: 0,
			Range: rag.Range{ByteStart: 0, ByteEnd: 12}, Text: "aaaaaaaaaaaa",
			ContentDigest: "d0", Chunker: "test",
		},
		{
			ID: "chunk-a1", DocumentID: "doc-a", Ordinal: 1,
			Range: rag.Range{ByteStart: 10, ByteEnd: 16}, Text: "aabbbb",
			ContentDigest: "d1", Chunker: "test",
		},
		{
			ID: "chunk-b0", DocumentID: "doc-b", Ordinal: 0,
			Range: rag.Range{ByteStart: 0, ByteEnd: 3}, Text: "ccc",
			ContentDigest: "d2", Chunker: "test",
		},
	}
}

// writeCorpus writes a corpus file and returns its path together with the
// digest CalculateID would compute for it.
func writeCorpus(t *testing.T, documents []rag.Document) (string, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "corpus.json")
	data, err := json.Marshal(documents)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(path, data, 0o644))
	corpusDigest, err := digest.JSON(documents)
	require.NoError(t, err)
	return path, corpusDigest
}

func testCorpus() []rag.Document {
	return []rag.Document{
		{
			ID: "doc-a", Title: "Document A", SourceURI: "https://example.test/a",
			Text: "aaaaaaaaaaaabbbb", ContentDigest: "c0",
		},
		{
			ID: "doc-b", Title: "Document B", SourceURI: "https://example.test/b",
			Text: "ccc", ContentDigest: "c1",
		},
	}
}

// A matching corpus supplies titles, and the rollups come from the bundle.
func TestInspectJoinsAMatchingCorpus(t *testing.T) {
	corpusPath, corpusDigest := writeCorpus(t, testCorpus())
	directory := writeBundle(t, Manifest{
		CorpusPath: corpusPath, CorpusDigest: corpusDigest, DocumentCount: 2,
	}, testChunks())

	inspection, err := Inspect(t.Context(), directory)
	require.NoError(t, err)
	require.Equal(t, CorpusMatches, inspection.Corpus.State)
	require.Empty(
		t, inspection.Corpus.Reason,
		"a matching corpus needs no explanation",
	)
	require.True(t, inspection.TitlesAvailable())
	require.Len(t, inspection.Chunks, 3)
	require.Len(t, inspection.Documents, 2)

	first := inspection.Documents[0]
	require.Equal(t, "doc-a", first.ID)
	require.Equal(t, "Document A", first.Title)
	require.Equal(t, 2, first.ChunkCount)

	// TotalRunes counts chunk text and Runes counts document text. Overlap
	// makes the first larger, and conflating the two would report a document as
	// bigger than it is.
	require.Equal(t, 18, first.TotalRunes)
	require.Equal(t, 16, first.Runes)
	require.Equal(t, 6, first.MinimumRunes)
	require.Equal(t, 12, first.MaximumRunes)
	require.Equal(t, 0, first.FirstOrdinal)
	require.Equal(t, 1, first.LastOrdinal)
}

// A corpus that is absent or has changed degrades the view. Neither refuses it.
func TestInspectDegradesRatherThanFailingWithoutACorpus(t *testing.T) {
	t.Run("absent", func(t *testing.T) {
		directory := writeBundle(t, Manifest{
			CorpusPath: filepath.Join(t.TempDir(), "gone.json"),
			CorpusDigest: "af65383f7284e39f1318a6633e8e7db0" +
				"b5c45c44424fe5e453398897412864e2",
			DocumentCount: 2,
		}, testChunks())

		inspection, err := Inspect(t.Context(), directory)
		require.NoError(t, err, "a missing corpus must not fail the read")
		require.Equal(t, CorpusAbsent, inspection.Corpus.State)
		require.Contains(t, inspection.Corpus.Reason, "cannot read")
		require.False(t, inspection.TitlesAvailable())
		require.Empty(t, inspection.Corpus.ActualDigest)

		// Every measurement that comes from the bundle survives.
		require.Len(t, inspection.Chunks, 3)
		require.Len(t, inspection.Documents, 2)
		require.Equal(t, 2, inspection.Documents[0].ChunkCount)
		require.Equal(t, 18, inspection.Documents[0].TotalRunes)
		require.Empty(t, inspection.Documents[0].Title)
		require.Equal(
			t, "doc-a", inspection.Documents[0].Name(),
			"a document with no title is named by its digest",
		)
	})

	t.Run("differs", func(t *testing.T) {
		changed := testCorpus()
		changed[0].Title = "Document A, retitled"
		corpusPath, actualDigest := writeCorpus(t, changed)
		directory := writeBundle(t, Manifest{
			CorpusPath: corpusPath, DocumentCount: 2,
			CorpusDigest: "0000000000000000000000000000000" +
				"00000000000000000000000000000000",
		}, testChunks())

		inspection, err := Inspect(t.Context(), directory)
		require.NoError(t, err, "a changed corpus must not fail the read")
		require.Equal(t, CorpusDiffers, inspection.Corpus.State)
		require.Contains(t, inspection.Corpus.Reason, "has changed since")
		require.Equal(t, actualDigest, inspection.Corpus.ActualDigest)
		require.NotEqual(
			t, inspection.Corpus.ExpectedDigest, inspection.Corpus.ActualDigest,
		)

		// The titles are still shown. They may be stale, and the reason says so;
		// hiding them would lose more than it protects.
		require.True(t, inspection.TitlesAvailable())
		require.Equal(t, "Document A, retitled", inspection.Documents[0].Title)
	})

	t.Run("no path recorded", func(t *testing.T) {
		directory := writeBundle(t, Manifest{DocumentCount: 2}, testChunks())
		inspection, err := Inspect(t.Context(), directory)
		require.NoError(t, err)
		require.Equal(t, CorpusAbsent, inspection.Corpus.State)
		require.Contains(t, inspection.Corpus.Reason, "no corpus path")
	})
}

// The digest is computed over the decoded documents, not over the file bytes.
// A corpus that was only reformatted must still read as matching, because
// CalculateID hashes the decoded values.
func TestInspectDigestsDecodedDocumentsNotFileBytes(t *testing.T) {
	documents := testCorpus()
	corpusDigest, err := digest.JSON(documents)
	require.NoError(t, err)

	indented, err := json.MarshalIndent(documents, "", "    ")
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "corpus.json")
	require.NoError(t, os.WriteFile(path, indented, 0o644))

	directory := writeBundle(t, Manifest{
		CorpusPath: path, CorpusDigest: corpusDigest, DocumentCount: 2,
	}, testChunks())
	inspection, err := Inspect(t.Context(), directory)
	require.NoError(t, err)
	require.Equal(
		t, CorpusMatches, inspection.Corpus.State,
		"reformatting a corpus must not read as changing it",
	)
}

// A chunks.json that disagrees with its manifest means one of the two was
// written by something other than Build, and every derived count would be
// silently wrong.
func TestInspectRefusesAChunkCountThatContradictsTheManifest(t *testing.T) {
	directory := writeBundle(t, Manifest{DocumentCount: 2}, testChunks())
	manifest, err := LoadManifest(directory)
	require.NoError(t, err)
	manifest.ChunkCount = 99
	data, err := json.Marshal(manifest)
	require.NoError(t, err)
	require.NoError(
		t, os.WriteFile(filepath.Join(directory, manifestName), data, 0o644),
	)

	_, err = Inspect(t.Context(), directory)
	require.ErrorContains(t, err, "holds 3 chunks but its manifest counts 99")
}

// Inspect must open no index. This is the guard that makes the whole decision
// worth taking: a reader that opens no index cannot corrupt one, and it cannot
// fail for want of a provider it will never call.
//
// The proof is that both artefacts are present and unreadable. Deleting them
// would only show that Inspect does not require them; making them unopenable
// shows that it does not touch them.
func TestInspectOpensNeitherTheVectorStoreNorTheLexicalIndex(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root bypasses the mode bits this test relies on")
	}
	corpusPath, corpusDigest := writeCorpus(t, testCorpus())
	directory := writeBundle(t, Manifest{
		CorpusPath: corpusPath, CorpusDigest: corpusDigest, DocumentCount: 2,
	}, testChunks())

	vectorPath := filepath.Join(directory, vectorName)
	require.NoError(t, os.WriteFile(vectorPath, []byte("not a database"), 0o000))
	blevePath := filepath.Join(directory, bleveName)
	require.NoError(t, os.Mkdir(blevePath, 0o755))
	require.NoError(t, os.WriteFile(
		filepath.Join(blevePath, "rag-manifest.json"), []byte("{}"), 0o000,
	))
	require.NoError(t, os.Chmod(blevePath, 0o000))
	t.Cleanup(func() { _ = os.Chmod(blevePath, 0o755) })

	// representations.json is also off limits. It is 2.7 MB in the real bundle
	// and holds a copy of every chunk's text.
	require.NoError(t, os.WriteFile(
		filepath.Join(directory, representationsName), []byte("["), 0o000,
	))

	inspection, err := Inspect(t.Context(), directory)
	require.NoError(t, err)
	require.Len(t, inspection.Chunks, 3)
	require.Equal(t, CorpusMatches, inspection.Corpus.State)
}

// ChunksOf sorts by ordinal rather than trusting file order. A screen that
// draws cut boundaries compares each chunk to its predecessor, so a wrong
// neighbour produces a wrong overlap marker.
func TestChunksOfReturnsOrdinalOrderWhateverTheFileOrderWas(t *testing.T) {
	shuffled := []rag.Chunk{
		{ID: "c2", DocumentID: "doc-a", Ordinal: 2, Text: "b"},
		{ID: "c0", DocumentID: "doc-a", Ordinal: 0, Text: "a"},
		{ID: "other", DocumentID: "doc-b", Ordinal: 0, Text: "z"},
		{ID: "c1", DocumentID: "doc-a", Ordinal: 1, Text: "c"},
	}
	directory := writeBundle(t, Manifest{DocumentCount: 2}, shuffled)
	inspection, err := Inspect(t.Context(), directory)
	require.NoError(t, err)

	chunks := inspection.ChunksOf("doc-a")
	require.Len(t, chunks, 3)
	require.Equal(t, []string{"c0", "c1", "c2"}, []string{
		chunks[0].ID, chunks[1].ID, chunks[2].ID,
	})

	// The rollup must also see all three regardless of order.
	document, ok := inspection.DocumentByID("doc-a")
	require.True(t, ok)
	require.Equal(t, 3, document.ChunkCount)
	require.Equal(t, 0, document.FirstOrdinal)
	require.Equal(t, 2, document.LastOrdinal)
}

// TestInspectsTheRealBundle asserts the exact counts the design doc measured.
// A rebuild that changes them must fail here so a person looks at why.
// Opt in with RAG_TTC_INDEX_BUNDLE=1.
func TestInspectsTheRealBundle(t *testing.T) {
	if os.Getenv("RAG_TTC_INDEX_BUNDLE") != "1" {
		t.Skip("set RAG_TTC_INDEX_BUNDLE=1 to read the real index bundle")
	}
	// t.Chdir, not os.Chdir. The working directory is process-global, and a
	// test that changes it permanently makes every later test in this package
	// depend on the order it ran in.
	t.Chdir(repoRoot(t))

	inspection, err := Inspect(t.Context(), realBundlePath)
	require.NoError(t, err)

	require.Equal(t, 1982, len(inspection.Chunks))
	require.Equal(t, 200, inspection.Manifest.DocumentCount)
	require.Equal(t, 200, len(inspection.Documents))
	require.Equal(t, "markdown-1200-120", inspection.Manifest.Chunker.Name)
	require.Equal(t, 1200, inspection.Manifest.Chunker.MaximumRunes)
	require.Equal(t, 120, inspection.Manifest.Chunker.OverlapRunes)

	// The join is exact: 200 chunk document ids against 200 corpus ids.
	require.Equal(
		t, CorpusMatches, inspection.Corpus.State, inspection.Corpus.Reason,
	)
	for _, document := range inspection.Documents {
		require.NotEmpty(
			t, document.Title, "document %s has no title", document.ID,
		)
	}

	// The document the design doc names, found by title.
	var arkansas Document
	for _, document := range inspection.Documents {
		if document.Title == "Arkansas Trees For Sale" {
			arkansas = document
		}
	}
	require.NotEmpty(t, arkansas.ID, "Arkansas Trees For Sale is not in the join")
	require.Equal(t, 19, arkansas.ChunkCount)
	require.Equal(t, 81, arkansas.MinimumRunes)
	require.Equal(t, 1200, arkansas.MaximumRunes)

	// The 110-rune chunk that reached a generation prompt.
	chunks := inspection.ChunksOf(arkansas.ID)
	require.Len(t, chunks, 19)
	require.Equal(t, "chunk-dfde14a88609c4b2", chunks[4].ID)
	require.Equal(t, 110, len([]rune(chunks[4].Text)))
	require.Equal(t, 1493, chunks[4].Range.ByteStart)
	require.Equal(t, 1603, chunks[4].Range.ByteEnd)
}

// A relative corpus path is resolved against the working directory, and a
// caller started elsewhere must be told that rather than shown a bare
// not-found for a file that exists.
func TestInspectNamesTheWorkingDirectoryForARelativeCorpus(t *testing.T) {
	directory := writeBundle(t, Manifest{
		CorpusPath:    filepath.Join("datasets", "ttc", "corpus.json"),
		DocumentCount: 2,
	}, testChunks())
	t.Chdir(t.TempDir())

	inspection, err := Inspect(t.Context(), directory)
	require.NoError(t, err)
	require.Equal(t, CorpusAbsent, inspection.Corpus.State)
	working, err := os.Getwd()
	require.NoError(t, err)
	require.Contains(t, inspection.Corpus.Reason, "resolved against")
	require.Contains(t, inspection.Corpus.Reason, working)
}
