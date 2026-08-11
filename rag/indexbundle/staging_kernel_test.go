package indexbundle

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func TestStagingKernelAdmitsBoundedValidatedRelations(t *testing.T) {
	document, chunk, representation, vector := stagingFixtures()
	kernel := openFixtureKernel(t, 2)

	require.NoError(t, kernel.addDocuments(t.Context(), []rag.Document{document}))
	require.NoError(t, kernel.addChunks(t.Context(), []rag.Chunk{chunk}))
	require.NoError(t, kernel.addRepresentations(t.Context(), []rag.Representation{representation}))
	require.NoError(t, kernel.addVectors(t.Context(), []rag.Vector{vector}))
	require.Equal(t, []int{1, 1, 1, 1}, []int{kernel.documents, kernel.chunks, kernel.reps, kernel.vectors})

	vector.Values[0] = 99
	var blob []byte
	require.NoError(t, kernel.db.QueryRow(`SELECT values_blob FROM vector WHERE representation_id = ?`, representation.ID).Scan(&blob))
	require.Equal(t, encodeStagedVector([]float32{1, 2}), blob, "staging must not retain caller-owned vector memory")
}

func TestStagingKernelFailsClosed(t *testing.T) {
	t.Run("transition", func(t *testing.T) {
		_, chunk, _, _ := stagingFixtures()
		kernel := openFixtureKernel(t, 2)
		require.ErrorContains(t, kernel.addChunks(t.Context(), []rag.Chunk{chunk}), "transition")
	})

	t.Run("batch bound", func(t *testing.T) {
		document, _, _, _ := stagingFixtures()
		kernel := openFixtureKernel(t, 1)
		require.ErrorContains(t, kernel.addDocuments(t.Context(), []rag.Document{document, document}), "outside")
	})

	t.Run("rollback", func(t *testing.T) {
		document, _, _, _ := stagingFixtures()
		kernel := openFixtureKernel(t, 2)
		require.NoError(t, kernel.addDocuments(t.Context(), []rag.Document{document}))
		err := kernel.addDocuments(t.Context(), []rag.Document{document})
		require.ErrorContains(t, err, "stage document")
		require.Equal(t, 1, kernel.documents)
		var count int
		require.NoError(t, kernel.db.QueryRow(`SELECT COUNT(*) FROM document`).Scan(&count))
		require.Equal(t, 1, count)
	})

	t.Run("lineage", func(t *testing.T) {
		document, chunk, _, _ := stagingFixtures()
		kernel := openFixtureKernel(t, 2)
		require.NoError(t, kernel.addDocuments(t.Context(), []rag.Document{document}))
		chunk.DocumentID = "missing"
		require.Error(t, kernel.addChunks(t.Context(), []rag.Chunk{chunk}))
		require.Equal(t, stagingDocuments, kernel.phase)
	})

	t.Run("vector identity", func(t *testing.T) {
		document, chunk, representation, vector := stagingFixtures()
		kernel := openFixtureKernel(t, 2)
		require.NoError(t, kernel.addDocuments(t.Context(), []rag.Document{document}))
		require.NoError(t, kernel.addChunks(t.Context(), []rag.Chunk{chunk}))
		require.NoError(t, kernel.addRepresentations(t.Context(), []rag.Representation{representation}))
		vector.Values[0] = float32(math.Inf(1))
		require.ErrorContains(t, kernel.addVectors(t.Context(), []rag.Vector{vector}), "finite")
		require.Equal(t, stagingRepresentations, kernel.phase)
	})

	t.Run("cancellation", func(t *testing.T) {
		document, _, _, _ := stagingFixtures()
		kernel := openFixtureKernel(t, 2)
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		require.ErrorIs(t, kernel.addDocuments(ctx, []rag.Document{document}), context.Canceled)
	})
}

func openFixtureKernel(t *testing.T, batchSize int) *stagingKernel {
	t.Helper()
	kernel, err := openStagingKernel(t.Context(), filepath.Join(t.TempDir(), "staging.sqlite"), stagingSpec{
		BatchSize: batchSize,
		Embedding: &VectorIdentity{Model: "fixture", Dimensions: 2},
	})
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, kernel.close()) })
	return kernel
}

func stagingFixtures() (rag.Document, rag.Chunk, rag.Representation, rag.Vector) {
	document := rag.Document{ID: "doc", Text: "hello", ContentDigest: digest.Text("hello")}
	chunk := rag.Chunk{
		ID: "chunk", DocumentID: document.ID, Ordinal: 0,
		Range: rag.Range{ByteStart: 0, ByteEnd: len(document.Text)},
		Text:  document.Text, ContentDigest: document.ContentDigest, Chunker: "fixture",
	}
	representation := rag.Representation{
		ID: "rep", ChunkID: chunk.ID, Kind: "raw",
		Text: chunk.Text, ContentDigest: chunk.ContentDigest,
	}
	vector := rag.Vector{RepresentationID: representation.ID, Model: "fixture", Values: []float32{1, 2}}
	return document, chunk, representation, vector
}
