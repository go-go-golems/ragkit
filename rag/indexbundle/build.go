package indexbundle

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/go-go-golems/ragkit/internal/fsutil"
	"github.com/go-go-golems/ragkit/rag"
	bleveindex "github.com/go-go-golems/ragkit/rag/lexical/bleve"
	"github.com/go-go-golems/ragkit/rag/vector/sqliteexact"
	"github.com/pkg/errors"
)

const (
	manifestName        = "manifest.json"
	chunksName          = "chunks.json"
	representationsName = "representations.json"
	bleveName           = "bleve"
	vectorName          = "vectors.sqlite"
)

func Build(ctx context.Context, input BuildInput) (BuildResult, error) {
	if err := validateBuildInput(input); err != nil {
		return BuildResult{}, err
	}
	lexicalTemplate := BackendIdentity{
		Backend: "bleve-bm25", Version: bleveindex.ManifestVersion, Channel: "bm25",
		TitleBoost: 2, BodyBoost: 1,
	}
	lexicalDigest, err := bleveindex.CalculateContentDigest(
		input.Documents, input.Chunks, input.Representations,
	)
	if err != nil {
		return BuildResult{}, errors.Wrap(err, "digest bundle lexical content")
	}
	lexicalTemplate.ContentDigest = lexicalDigest
	vectorIdentity := cloneVectorIdentity(input.Embedding)
	if vectorIdentity != nil {
		vectorIdentity.ContentDigest, err = sqliteexact.CalculateContentDigest(
			input.Representations, input.Chunks, input.Vectors,
		)
		if err != nil {
			return BuildResult{}, errors.Wrap(err, "digest bundle vector content")
		}
	}
	bundleID, corpusDigest, representationDigest, kinds, err := CalculateID(
		input.Documents, input.Representations, input.Chunker,
		lexicalTemplate, vectorIdentity,
	)
	if err != nil {
		return BuildResult{}, err
	}
	if vectorIdentity != nil {
		vectorIdentity.RepresentationDigest = representationDigest
	}
	manifest := Manifest{
		SchemaVersion: SchemaVersion, BundleID: bundleID,
		CreatedAt: time.Now().UTC(), CorpusDigest: corpusDigest,
		CorpusPath: input.CorpusPath, DocumentCount: len(input.Documents),
		ChunkCount: len(input.Chunks), RepresentationCount: len(input.Representations),
		Chunker: input.Chunker, RepresentationKinds: kinds,
		Lexical: lexicalTemplate, Vector: vectorIdentity,
	}
	finalPath := filepath.Join(input.OutputRoot, bundleID)
	if _, statErr := os.Stat(finalPath); statErr == nil {
		existing, loadErr := LoadManifest(finalPath)
		if loadErr != nil {
			return BuildResult{}, errors.Wrap(loadErr, "existing bundle is invalid")
		}
		if existing.BundleID != bundleID {
			return BuildResult{}, errors.Errorf(
				"existing bundle ID %q differs from expected %q",
				existing.BundleID, bundleID,
			)
		}
		data, validateErr := loadData(finalPath, existing)
		if validateErr != nil {
			return BuildResult{}, errors.Wrap(validateErr, "existing bundle is incomplete")
		}
		if validateErr := validateStoredIdentity(existing, data); validateErr != nil {
			return BuildResult{}, errors.Wrap(validateErr, "existing bundle identity is invalid")
		}
		return measureResult(ctx, finalPath, existing, true)
	} else if !os.IsNotExist(statErr) {
		return BuildResult{}, errors.Wrap(statErr, "inspect bundle destination")
	}
	if err := os.MkdirAll(input.OutputRoot, 0o700); err != nil {
		return BuildResult{}, errors.Wrap(err, "create bundle output root")
	}
	temporary, err := os.MkdirTemp(input.OutputRoot, ".bundle-partial-*")
	if err != nil {
		return BuildResult{}, errors.Wrap(err, "create temporary bundle")
	}
	published := false
	defer func() {
		if !published {
			_ = os.RemoveAll(temporary)
		}
	}()
	if err := writeJSON(ctx, filepath.Join(temporary, chunksName), input.Chunks); err != nil {
		return BuildResult{}, err
	}
	if err := writeJSON(ctx, filepath.Join(temporary, representationsName), input.Representations); err != nil {
		return BuildResult{}, err
	}
	lexical, lexicalManifest, err := bleveindex.Build(ctx, bleveindex.Config{
		Path: filepath.Join(temporary, bleveName), Channel: "bm25",
	}, input.Documents, input.Chunks, input.Representations)
	if err != nil {
		return BuildResult{}, errors.Wrap(err, "build bundle lexical index")
	}
	if err := lexical.Close(); err != nil {
		return BuildResult{}, errors.Wrap(err, "close bundle lexical index")
	}
	manifest.Lexical = lexicalIdentity(lexicalManifest)
	if manifest.Lexical.ContentDigest != lexicalTemplate.ContentDigest {
		return BuildResult{}, errors.New("built lexical content digest differs from planned identity")
	}
	if vectorIdentity != nil {
		vector, vectorErr := sqliteexact.Build(ctx, sqliteexact.Config{
			Path:  filepath.Join(temporary, vectorName),
			Model: vectorIdentity.Model, Channel: vectorIdentity.Channel,
		}, input.Representations, input.Chunks, input.Vectors, input.QueryEmbedder)
		if vectorErr != nil {
			return BuildResult{}, errors.Wrap(vectorErr, "build bundle vector index")
		}
		if err := vector.Close(); err != nil {
			return BuildResult{}, errors.Wrap(err, "close bundle vector index")
		}
		persisted, inspectErr := sqliteexact.Inspect(filepath.Join(temporary, vectorName))
		if inspectErr != nil {
			return BuildResult{}, errors.Wrap(inspectErr, "inspect built bundle vector index")
		}
		if persisted.ContentDigest != vectorIdentity.ContentDigest {
			return BuildResult{}, errors.New("built vector content digest differs from planned identity")
		}
	}
	if err := writeJSON(ctx, filepath.Join(temporary, manifestName), manifest); err != nil {
		return BuildResult{}, err
	}
	if err := fsutil.SyncDirectory(temporary); err != nil {
		return BuildResult{}, errors.Wrap(err, "sync temporary bundle")
	}
	if err := os.Rename(temporary, finalPath); err != nil {
		return BuildResult{}, errors.Wrap(err, "publish bundle")
	}
	published = true
	if err := fsutil.SyncDirectory(input.OutputRoot); err != nil {
		return BuildResult{}, errors.Wrap(err, "sync bundle output root")
	}
	return measureResult(ctx, finalPath, manifest, false)
}

func validateBuildInput(input BuildInput) error {
	if strings.TrimSpace(input.OutputRoot) == "" {
		return errors.New("bundle output root is required")
	}
	if len(input.Documents) == 0 || len(input.Chunks) == 0 || len(input.Representations) == 0 {
		return errors.New("bundle documents, chunks, and representations are required")
	}
	if err := rag.ValidateCorpus(input.Documents, input.Chunks); err != nil {
		return errors.Wrap(err, "validate bundle corpus")
	}
	if err := rag.ValidateRepresentations(input.Chunks, input.Representations); err != nil {
		return errors.Wrap(err, "validate bundle representations")
	}
	if strings.TrimSpace(input.Chunker.Name) == "" || input.Chunker.MaximumRunes <= 0 ||
		input.Chunker.OverlapRunes < 0 || input.Chunker.OverlapRunes >= input.Chunker.MaximumRunes {
		return errors.New("valid bundle chunker identity is required")
	}
	// A nil Embedding identity requests a lexical-only bundle; vector inputs
	// must then be absent so a half-configured build fails loudly.
	if input.Embedding == nil {
		if input.QueryEmbedder != nil || len(input.Vectors) > 0 {
			return errors.New("vector inputs supplied without an embedding identity")
		}
		return nil
	}
	if input.QueryEmbedder == nil || len(input.Vectors) != len(input.Representations) {
		return errors.New("query embedder and one vector per representation are required")
	}
	if strings.TrimSpace(input.Embedding.Provider) == "" ||
		strings.TrimSpace(input.Embedding.Model) == "" || input.Embedding.Dimensions <= 0 {
		return errors.New("valid vector embedding provider, model, and dimensions are required")
	}
	for _, vector := range input.Vectors {
		if vector.Model != input.Embedding.Model || len(vector.Values) != input.Embedding.Dimensions {
			return errors.New("vector data differs from embedding identity")
		}
	}
	return nil
}

func writeJSON(ctx context.Context, path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return errors.Wrap(err, "encode bundle JSON")
	}
	data = append(data, '\n')
	return errors.Wrap(
		fsutil.AtomicWrite(ctx, path, data, fsutil.AtomicWriteOptions{}),
		"write bundle JSON",
	)
}

func measureResult(ctx context.Context, path string, manifest Manifest, reused bool) (BuildResult, error) {
	bleveBytes, err := fsutil.DirectorySize(ctx, filepath.Join(path, bleveName))
	if err != nil {
		return BuildResult{}, errors.Wrap(err, "measure bundle lexical index")
	}
	var vectorBytes int64
	if manifest.Vector != nil {
		vectorInfo, err := os.Stat(filepath.Join(path, vectorName))
		if err != nil {
			return BuildResult{}, errors.Wrap(err, "measure bundle vector index")
		}
		vectorBytes = vectorInfo.Size()
	}
	return BuildResult{
		Manifest: manifest, Path: path, Reused: reused,
		BleveBytes: bleveBytes, VectorBytes: vectorBytes,
	}, nil
}
