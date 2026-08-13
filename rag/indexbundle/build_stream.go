package indexbundle

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/go-go-golems/ragkit/internal/fsutil"
	"github.com/go-go-golems/ragkit/rag"
	contentsqlite "github.com/go-go-golems/ragkit/rag/content/sqlite"
	bleveindex "github.com/go-go-golems/ragkit/rag/lexical/bleve"
	"github.com/go-go-golems/ragkit/rag/vector/sqliteexact"
	"github.com/pkg/errors"
)

const stagingName = "staging.sqlite"

// BuildStream constructs and atomically publishes an immutable bundle without
// requiring the complete vector collection to exist in Go memory. All caller
// input crosses the Stager's bounded, validated admission boundary.
func BuildStream(ctx context.Context, input StreamInput) (BuildResult, error) {
	if err := validateStreamInput(ctx, input); err != nil {
		return BuildResult{}, err
	}
	observeStreamStage(input, BuildStageInputValidated)
	if err := os.MkdirAll(input.OutputRoot, 0o700); err != nil {
		return BuildResult{}, errors.Wrap(err, "create bundle output root")
	}
	temporary, err := os.MkdirTemp(input.OutputRoot, ".bundle-partial-*")
	if err != nil {
		return BuildResult{}, errors.Wrap(err, "create temporary streamed bundle")
	}
	published := false
	defer func() {
		if !published {
			_ = os.RemoveAll(temporary)
		}
	}()
	observeStreamStage(input, BuildStageTemporaryCreated)

	stagingPath := filepath.Join(temporary, stagingName)
	kernel, err := openStagingKernel(ctx, stagingPath, stagingSpec{
		BatchSize: input.BatchSize, Chunker: input.Chunker, Embedding: cloneVectorIdentity(input.Embedding),
	})
	if err != nil {
		return BuildResult{}, err
	}
	kernelClosed := false
	defer func() {
		if !kernelClosed {
			_ = kernel.close()
		}
	}()
	if err := input.Produce(ctx, &Stager{kernel: kernel}); err != nil {
		return BuildResult{}, errors.Wrap(err, "produce streamed bundle input")
	}
	observeStreamStage(input, BuildStageStagingProduced)
	plan, err := kernel.seal(ctx)
	if err != nil {
		return BuildResult{}, errors.Wrap(err, "seal streamed bundle input")
	}
	observeStreamStage(input, BuildStageStagingSealed)
	observeStreamStage(input, BuildStageIdentityPlanned)

	finalPath := filepath.Join(input.OutputRoot, plan.BundleID)
	if _, statErr := os.Stat(finalPath); statErr == nil {
		observeStreamStage(input, BuildStageExistingVerificationStarted)
		if err := kernel.close(); err != nil {
			return BuildResult{}, errors.Wrap(err, "close streamed staging database")
		}
		kernelClosed = true
		existing, err := verifyExistingStreamedBundle(ctx, finalPath, plan.BundleID, input.CorpusPath, input.ScratchDirectory)
		if err != nil {
			return BuildResult{}, err
		}
		observeStreamStage(input, BuildStageExistingVerified)
		result, err := measureResult(ctx, finalPath, existing, true)
		if err == nil {
			observeStreamStage(input, BuildStageResultMeasured)
		}
		return result, err
	} else if !os.IsNotExist(statErr) {
		return BuildResult{}, errors.Wrap(statErr, "inspect streamed bundle destination")
	}
	observeStreamStage(input, BuildStageDestinationAbsent)

	manifest := Manifest{
		SchemaVersion: SchemaVersion, BundleID: plan.BundleID, CreatedAt: time.Now().UTC(),
		CorpusDigest: plan.CorpusDigest, ChunkDigest: plan.ChunkDigest, CorpusPath: input.CorpusPath,
		DocumentCount: plan.DocumentCount, ChunkCount: plan.ChunkCount,
		RepresentationCount: plan.RepresentationCount, Chunker: input.Chunker,
		RepresentationKinds: plan.RepresentationKinds, Lexical: plan.Lexical,
		Vector:  cloneVectorIdentity(plan.Vector),
		Content: cloneContentIdentity(plan.Content),
	}
	if err := kernel.writeJSONArray(ctx, filepath.Join(temporary, chunksName), "chunk"); err != nil {
		return BuildResult{}, err
	}
	if err := kernel.writeJSONArray(ctx, filepath.Join(temporary, representationsName), "representation"); err != nil {
		return BuildResult{}, err
	}
	observeStreamStage(input, BuildStagePayloadsWritten)
	contentResult, err := contentsqlite.Build(ctx, contentsqlite.BuildInput{
		Path: filepath.Join(temporary, contentName),
		Documents: func(ctx context.Context, yield func(rag.Document) error) error {
			return kernel.produceDocuments(ctx, yield)
		},
		Chunks: func(ctx context.Context, yield func(rag.Chunk) error) error {
			return kernel.produceChunks(ctx, yield)
		},
	})
	if err != nil {
		return BuildResult{}, errors.Wrap(err, "build streamed content store")
	}
	if plan.Content == nil || contentResult.Identity != *plan.Content {
		return BuildResult{}, errors.New("streamed content identity differs from sealed identity")
	}
	manifest.Content = cloneContentIdentity(&contentResult.Identity)
	observeStreamStage(input, BuildStageContentBuilt)

	lexical, lexicalManifest, err := bleveindex.BuildRecords(ctx, bleveindex.Config{
		Path: filepath.Join(temporary, bleveName), Channel: plan.Lexical.Channel,
		TitleBoost: plan.Lexical.TitleBoost, BodyBoost: plan.Lexical.BodyBoost,
		BatchSize: input.BatchSize,
	}, plan.RepresentationCount, func(yield func(bleveindex.Record) error) error {
		return kernel.produceLexical(ctx, yield)
	})
	if err != nil {
		return BuildResult{}, errors.Wrap(err, "build streamed lexical index")
	}
	if err := lexical.Close(); err != nil {
		return BuildResult{}, errors.Wrap(err, "close streamed lexical index")
	}
	manifest.Lexical = lexicalIdentity(lexicalManifest)
	if manifest.Lexical.ContentDigest != plan.Lexical.ContentDigest {
		return BuildResult{}, errors.New("streamed lexical content digest differs from sealed identity")
	}
	observeStreamStage(input, BuildStageLexicalBuilt)

	if plan.Vector != nil {
		vector, err := sqliteexact.BuildEntries(ctx, sqliteexact.Config{
			Path: filepath.Join(temporary, vectorName), Model: plan.Vector.Model, Channel: plan.Vector.Channel,
		}, plan.RepresentationCount, func(yield func(sqliteexact.Entry) error) error {
			return kernel.produceVectors(ctx, yield)
		}, input.QueryEmbedder)
		if err != nil {
			return BuildResult{}, errors.Wrap(err, "build streamed vector index")
		}
		if err := vector.Close(); err != nil {
			return BuildResult{}, errors.Wrap(err, "close streamed vector index")
		}
		persisted, err := sqliteexact.InspectContext(ctx, filepath.Join(temporary, vectorName))
		if err != nil {
			return BuildResult{}, errors.Wrap(err, "inspect streamed vector index")
		}
		if persisted.ContentDigest != plan.Vector.ContentDigest {
			return BuildResult{}, errors.New("streamed vector content digest differs from sealed identity")
		}
		observeStreamStage(input, BuildStageVectorBuilt)
	}

	if err := kernel.close(); err != nil {
		return BuildResult{}, errors.Wrap(err, "close streamed staging database")
	}
	kernelClosed = true
	if err := os.Remove(stagingPath); err != nil {
		return BuildResult{}, errors.Wrap(err, "remove streamed staging database")
	}
	if err := writeJSON(ctx, filepath.Join(temporary, manifestName), manifest); err != nil {
		return BuildResult{}, err
	}
	observeStreamStage(input, BuildStageManifestWritten)
	if err := fsutil.SyncDirectory(temporary); err != nil {
		return BuildResult{}, errors.Wrap(err, "sync temporary streamed bundle")
	}
	if err := os.Rename(temporary, finalPath); err != nil {
		return BuildResult{}, errors.Wrap(err, "publish streamed bundle")
	}
	published = true
	observeStreamStage(input, BuildStageBundlePublished)
	if err := fsutil.SyncDirectory(input.OutputRoot); err != nil {
		return BuildResult{}, errors.Wrap(err, "sync streamed bundle output root")
	}
	result, err := measureResult(ctx, finalPath, manifest, false)
	if err == nil {
		observeStreamStage(input, BuildStageResultMeasured)
	}
	return result, err
}

func validateStreamInput(ctx context.Context, input StreamInput) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if strings.TrimSpace(input.OutputRoot) == "" || strings.TrimSpace(input.CorpusPath) == "" {
		return errors.New("streamed bundle output root and corpus path are required")
	}
	if strings.TrimSpace(input.ScratchDirectory) == "" {
		return errors.New("streamed bundle scratch directory is required")
	}
	if input.BatchSize < 1 || input.Produce == nil {
		return errors.New("streamed bundle batch size and producer are required")
	}
	if strings.TrimSpace(input.Chunker.Name) == "" || input.Chunker.MaximumRunes <= 0 ||
		input.Chunker.OverlapRunes < 0 || input.Chunker.OverlapRunes >= input.Chunker.MaximumRunes {
		return errors.New("valid streamed bundle chunker identity is required")
	}
	if input.Embedding == nil {
		if input.QueryEmbedder != nil {
			return errors.New("streamed query embedder supplied without an embedding identity")
		}
		return nil
	}
	if input.QueryEmbedder == nil || strings.TrimSpace(input.Embedding.Provider) == "" ||
		strings.TrimSpace(input.Embedding.Model) == "" || input.Embedding.Dimensions <= 0 {
		return errors.New("complete streamed embedding identity and query embedder are required")
	}
	if input.Embedding.Backend != "sqlite-exact" || input.Embedding.Version != 1 ||
		strings.TrimSpace(input.Embedding.Channel) == "" {
		return errors.New("streamed vector identity must declare sqlite-exact backend version 1 and a channel")
	}
	return nil
}

func verifyExistingStreamedBundle(ctx context.Context, path, bundleID, corpusPath, scratchDirectory string) (Manifest, error) {
	manifest, err := Verify(ctx, VerifyOptions{
		Path: path, ExpectedBundleID: bundleID, ExpectedCorpusPath: corpusPath,
		ScratchDirectory: scratchDirectory,
	})
	if err != nil {
		return Manifest{}, errors.Wrap(err, "existing streamed bundle is invalid")
	}
	return manifest, nil
}

func observeStreamStage(input StreamInput, stage BuildStage) {
	if input.ObserveStage != nil {
		input.ObserveStage(stage)
	}
}
