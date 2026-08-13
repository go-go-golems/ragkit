package indexbundle

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/require"
)

func TestBuildStreamMatchesEagerIdentityAndOpens(t *testing.T) {
	eagerInput := fixtureInput(t, filepath.Join(t.TempDir(), "eager"))
	eager, err := Build(t.Context(), eagerInput)
	require.NoError(t, err)

	streamInput := streamFixture(eagerInput, filepath.Join(t.TempDir(), "streamed"), 1)
	var stages []BuildStage
	streamInput.ObserveStage = func(stage BuildStage) { stages = append(stages, stage) }
	streamed, err := BuildStream(t.Context(), streamInput)
	require.NoError(t, err)
	require.Equal(t, eager.Manifest.BundleID, streamed.Manifest.BundleID)
	require.Equal(t, eager.Manifest.CorpusDigest, streamed.Manifest.CorpusDigest)
	require.Equal(t, eager.Manifest.ChunkDigest, streamed.Manifest.ChunkDigest)
	require.Equal(t, eager.Manifest.Lexical, streamed.Manifest.Lexical)
	require.Equal(t, eager.Manifest.Vector, streamed.Manifest.Vector)
	require.Equal(t, eager.Manifest.Content, streamed.Manifest.Content)
	require.NoFileExists(t, filepath.Join(streamed.Path, stagingName))
	require.Equal(t, []BuildStage{
		BuildStageInputValidated,
		BuildStageTemporaryCreated,
		BuildStageStagingProduced,
		BuildStageStagingSealed,
		BuildStageIdentityPlanned,
		BuildStageDestinationAbsent,
		BuildStagePayloadsWritten,
		BuildStageContentBuilt,
		BuildStageLexicalBuilt,
		BuildStageVectorBuilt,
		BuildStageManifestWritten,
		BuildStageBundlePublished,
		BuildStageResultMeasured,
	}, stages)

	bundle, err := Open(t.Context(), OpenOptions{
		Path: streamed.Path, QueryEmbedder: eagerInput.QueryEmbedder,
		EmbeddingProvider: "hash", EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 16,
		ScratchDirectory: filepath.Join(streamInput.OutputRoot, "tmp"),
	})
	require.NoError(t, err)
	hits, err := bundle.Lexical.Search(t.Context(), rag.Query{Text: "lobed leaves"}, 5)
	require.NoError(t, err)
	require.NotEmpty(t, hits)
	require.NoError(t, bundle.Close())

	stages = nil
	reused, err := BuildStream(t.Context(), streamInput)
	require.NoError(t, err)
	require.True(t, reused.Reused)
	require.Equal(t, streamed.Manifest.BundleID, reused.Manifest.BundleID)
}

func TestBuildStreamRejectsProducerFailureWithoutPublishing(t *testing.T) {
	eager := fixtureInput(t, filepath.Join(t.TempDir(), "unused"))
	root := filepath.Join(t.TempDir(), "streamed")
	valid := streamFixture(eager, root, 1)
	input := valid
	input.Produce = func(ctx context.Context, stager *Stager) error {
		return errors.New("producer stopped")
	}
	_, err := BuildStream(t.Context(), input)
	require.ErrorContains(t, err, "producer stopped")
	entries, readErr := os.ReadDir(root)
	require.NoError(t, readErr)
	require.Empty(t, entries)

	recovered, err := BuildStream(t.Context(), valid)
	require.NoError(t, err)
	require.False(t, recovered.Reused)
	require.DirExists(t, recovered.Path)
}

func streamFixture(eager BuildInput, outputRoot string, batchSize int) StreamInput {
	return StreamInput{
		OutputRoot: outputRoot, CorpusPath: eager.CorpusPath, Chunker: eager.Chunker,
		Embedding: cloneVectorIdentity(eager.Embedding), QueryEmbedder: eager.QueryEmbedder,
		BatchSize:        batchSize,
		ScratchDirectory: filepath.Join(outputRoot, "tmp"),
		Produce: func(ctx context.Context, stager *Stager) error {
			if err := addBatches(ctx, eager.Documents, batchSize, stager.AddDocuments); err != nil {
				return err
			}
			if err := addBatches(ctx, eager.Chunks, batchSize, stager.AddChunks); err != nil {
				return err
			}
			if err := addBatches(ctx, eager.Representations, batchSize, stager.AddRepresentations); err != nil {
				return err
			}
			return addBatches(ctx, eager.Vectors, batchSize, stager.AddVectors)
		},
	}
}

func addBatches[T any](ctx context.Context, values []T, size int, add func(context.Context, []T) error) error {
	for start := 0; start < len(values); start += size {
		end := min(start+size, len(values))
		if err := add(ctx, values[start:end]); err != nil {
			return err
		}
	}
	return nil
}
