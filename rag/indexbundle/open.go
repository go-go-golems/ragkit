package indexbundle

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/internal/jsonutil"
	"github.com/go-go-golems/ragkit/rag"
	bleveindex "github.com/go-go-golems/ragkit/rag/lexical/bleve"
	"github.com/go-go-golems/ragkit/rag/vector/sqliteexact"
	"github.com/pkg/errors"
)

func LoadManifest(path string) (Manifest, error) {
	var manifest Manifest
	if err := readJSON(filepath.Join(path, manifestName), &manifest); err != nil {
		return Manifest{}, err
	}
	if manifest.SchemaVersion != SchemaVersion {
		return Manifest{}, errors.Errorf(
			"unsupported bundle schema version %d", manifest.SchemaVersion,
		)
	}
	if strings.TrimSpace(manifest.BundleID) == "" {
		return Manifest{}, errors.New("bundle manifest has no bundle ID")
	}
	if manifest.DocumentCount < 0 || manifest.ChunkCount < 0 || manifest.RepresentationCount < 0 {
		return Manifest{}, errors.New("bundle manifest counts must be non-negative")
	}
	return manifest, nil
}

func Open(ctx context.Context, options OpenOptions) (*Bundle, error) {
	verified, err := loadVerifiedBundle(ctx, options.Path)
	if err != nil {
		return nil, err
	}
	manifest := verified.manifest
	// Lexical-only bundles (no vector identity) are a supported serving and
	// rollback configuration: they open without an embedder and expose a nil
	// Vector index.
	if manifest.Vector != nil {
		if options.QueryEmbedder == nil {
			return nil, errors.New("query embedder is required")
		}
		if options.EmbeddingProvider != manifest.Vector.Provider {
			return nil, errors.Errorf(
				"query embedding provider %q differs from bundle provider %q",
				options.EmbeddingProvider, manifest.Vector.Provider,
			)
		}
		if options.EmbeddingModel != manifest.Vector.Model {
			return nil, errors.Errorf(
				"query embedding model %q differs from bundle model %q",
				options.EmbeddingModel, manifest.Vector.Model,
			)
		}
		if options.EmbeddingDimensions != manifest.Vector.Dimensions {
			return nil, errors.Errorf(
				"query embedding dimensions %d differ from bundle dimensions %d",
				options.EmbeddingDimensions, manifest.Vector.Dimensions,
			)
		}
	}
	lexical, err := bleveindex.Open(
		filepath.Join(options.Path, bleveName), manifest.Lexical.Channel,
	)
	if err != nil {
		return nil, errors.Wrap(err, "open bundle lexical index")
	}
	var vector rag.Index
	if manifest.Vector != nil {
		vector, err = sqliteexact.Open(
			filepath.Join(options.Path, vectorName),
			manifest.Vector.Model, manifest.Vector.Channel, options.QueryEmbedder,
		)
		if err != nil {
			_ = lexical.Close()
			return nil, errors.Wrap(err, "open bundle vector index")
		}
	}
	return &Bundle{
		Manifest: manifest, Chunks: verified.chunks,
		Representations: verified.representations,
		Lexical:         lexical, Vector: vector,
	}, nil
}

type verifiedManifest struct {
	path     string
	manifest Manifest
}

type verifiedChunks struct {
	verifiedManifest
	chunks []rag.Chunk
}

type verifiedData struct {
	verifiedChunks
	representations []rag.Representation
}

func loadVerifiedManifest(ctx context.Context, path string) (verifiedManifest, error) {
	if err := ctx.Err(); err != nil {
		return verifiedManifest{}, err
	}
	manifest, err := LoadManifest(path)
	if err != nil {
		return verifiedManifest{}, errors.Wrap(err, "load index bundle manifest")
	}
	return verifiedManifest{path: path, manifest: manifest}, nil
}

func loadVerifiedChunks(ctx context.Context, state verifiedManifest) (verifiedChunks, error) {
	if err := ctx.Err(); err != nil {
		return verifiedChunks{}, err
	}
	var chunks []rag.Chunk
	if err := readJSON(filepath.Join(state.path, chunksName), &chunks); err != nil {
		return verifiedChunks{}, errors.Wrap(err, "load bundle chunks")
	}
	if len(chunks) != state.manifest.ChunkCount {
		return verifiedChunks{}, errors.Errorf(
			"bundle data counts differ from manifest: holds %d chunks but manifest counts %d",
			len(chunks), state.manifest.ChunkCount,
		)
	}
	if err := validateStoredChunks(chunks, state.manifest.DocumentCount); err != nil {
		return verifiedChunks{}, errors.Wrap(err, "validate bundle chunks")
	}
	chunkDigest, err := digest.JSON(chunks)
	if err != nil {
		return verifiedChunks{}, errors.Wrap(err, "digest bundle chunks")
	}
	if chunkDigest != state.manifest.ChunkDigest {
		return verifiedChunks{}, errors.New("bundle chunk digest differs from manifest")
	}
	if err := ctx.Err(); err != nil {
		return verifiedChunks{}, err
	}
	return verifiedChunks{verifiedManifest: state, chunks: chunks}, nil
}

func loadVerifiedData(ctx context.Context, state verifiedChunks) (verifiedData, error) {
	verified, err := loadVerifiedStoredData(ctx, state)
	if err != nil {
		return verifiedData{}, err
	}
	if err := validateBackendIdentity(ctx, verified); err != nil {
		return verifiedData{}, err
	}
	return verified, nil
}

// loadVerifiedStoredData validates the manifest, chunks, representations, and
// their content-derived bundle identity without opening either search backend.
// Callers that only need source lineage can therefore run while a serving
// Bundle already owns the backend's process-local/exclusive file lock.
func loadVerifiedStoredData(ctx context.Context, state verifiedChunks) (verifiedData, error) {
	if err := ctx.Err(); err != nil {
		return verifiedData{}, err
	}
	manifest := state.manifest
	path := state.path
	var representations []rag.Representation
	if err := readJSON(filepath.Join(path, representationsName), &representations); err != nil {
		return verifiedData{}, errors.Wrap(err, "load bundle representations")
	}
	if len(representations) != manifest.RepresentationCount {
		return verifiedData{}, errors.New("bundle representation count differs from manifest")
	}
	if err := rag.ValidateRepresentations(state.chunks, representations); err != nil {
		return verifiedData{}, errors.Wrap(err, "validate bundle representations")
	}
	if err := ctx.Err(); err != nil {
		return verifiedData{}, err
	}
	verified := verifiedData{verifiedChunks: state, representations: representations}
	if err := validateStoredIdentity(verified); err != nil {
		return verifiedData{}, err
	}
	return verified, nil
}

func loadVerifiedBundle(ctx context.Context, path string) (verifiedData, error) {
	manifest, err := loadVerifiedManifest(ctx, path)
	if err != nil {
		return verifiedData{}, err
	}
	chunks, err := loadVerifiedChunks(ctx, manifest)
	if err != nil {
		return verifiedData{}, err
	}
	return loadVerifiedData(ctx, chunks)
}

func validateStoredIdentity(data verifiedData) error {
	manifest := data.manifest
	chunkDigest, err := digest.JSON(data.chunks)
	if err != nil {
		return err
	}
	representationDigest, err := digest.JSON(data.representations)
	if err != nil {
		return err
	}
	kinds := representationKinds(data.representations)
	expectedID, err := calculateIDFromDigests(
		manifest.CorpusDigest, chunkDigest, representationDigest, kinds,
		manifest.Chunker, manifest.Lexical, manifest.Vector,
	)
	if err != nil {
		return err
	}
	if expectedID != manifest.BundleID || chunkDigest != manifest.ChunkDigest || !reflect.DeepEqual(kinds, manifest.RepresentationKinds) {
		return errors.New("bundle identity does not match stored data")
	}
	if manifest.Vector != nil {
		if manifest.Vector.RepresentationDigest != representationDigest {
			return errors.New("bundle vector representation digest mismatch")
		}
	}
	return nil
}

func validateBackendIdentity(ctx context.Context, data verifiedData) error {
	if err := validateLexicalBackendIdentity(ctx, data.verifiedManifest); err != nil {
		return err
	}
	return validateVectorBackendIdentity(ctx, data.verifiedManifest)
}

func validateLexicalBackendIdentity(ctx context.Context, data verifiedManifest) error {
	manifest := data.manifest
	path := data.path
	bleveData, err := os.ReadFile(filepath.Join(path, bleveName, "rag-manifest.json"))
	if err != nil {
		return errors.Wrap(err, "read bundle lexical manifest")
	}
	lexicalManifest, err := jsonutil.DecodeStrict[bleveindex.Manifest](bleveData)
	if err != nil {
		return errors.Wrap(err, "decode bundle lexical manifest")
	}
	if lexicalIdentity(lexicalManifest) != manifest.Lexical ||
		lexicalManifest.RepresentationCount != manifest.RepresentationCount {
		return errors.New("bundle lexical identity differs from manifest")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	lexicalDigest, err := bleveindex.InspectContentDigest(ctx, filepath.Join(path, bleveName), 512)
	if err != nil {
		return errors.Wrap(err, "inspect bundle lexical content")
	}
	if lexicalDigest != manifest.Lexical.ContentDigest {
		return errors.New("bundle lexical content differs from manifest")
	}
	return nil
}

func validateVectorBackendIdentity(ctx context.Context, data verifiedManifest) error {
	manifest := data.manifest
	path := data.path
	if manifest.Vector != nil {
		if err := ctx.Err(); err != nil {
			return err
		}
		vectorManifest, err := sqliteexact.Inspect(filepath.Join(path, vectorName))
		if err != nil {
			return errors.Wrap(err, "inspect bundle vector index")
		}
		if vectorManifest.Model != manifest.Vector.Model ||
			vectorManifest.Dimensions != manifest.Vector.Dimensions ||
			vectorManifest.RepresentationCount != manifest.RepresentationCount ||
			vectorManifest.ContentDigest != manifest.Vector.ContentDigest {
			return errors.New("bundle vector identity differs from manifest")
		}
	}
	// Corpus identity cannot be reconstructed from chunks because overlap and
	// Markdown boundaries duplicate text. Persist the digest check through the
	// manifest and bundle ID; callers validate source corpus when building.
	return ctx.Err()
}

func validateStoredChunks(chunks []rag.Chunk, documentCount int) error {
	documentIDs := make(map[string]struct{})
	chunkIDs := make(map[string]struct{}, len(chunks))
	ordinals := make(map[string]map[int]struct{}, documentCount)
	for _, chunk := range chunks {
		if chunk.ID == "" || chunk.DocumentID == "" || chunk.ContentDigest == "" {
			return errors.New("bundle contains an invalid chunk identity")
		}
		if digest.Text(chunk.Text) != chunk.ContentDigest {
			return errors.Errorf("bundle chunk %q content digest mismatch", chunk.ID)
		}
		if chunk.Range.ByteStart < 0 || chunk.Range.ByteEnd < chunk.Range.ByteStart ||
			chunk.Range.ByteEnd-chunk.Range.ByteStart != len([]byte(chunk.Text)) {
			return errors.Errorf("bundle chunk %q has an invalid stored byte range", chunk.ID)
		}
		if _, duplicate := chunkIDs[chunk.ID]; duplicate {
			return errors.Errorf("bundle contains duplicate chunk %q", chunk.ID)
		}
		if chunk.Ordinal < 0 {
			return errors.Errorf("bundle chunk %q has negative ordinal %d", chunk.ID, chunk.Ordinal)
		}
		if ordinals[chunk.DocumentID] == nil {
			ordinals[chunk.DocumentID] = map[int]struct{}{}
		}
		if _, duplicate := ordinals[chunk.DocumentID][chunk.Ordinal]; duplicate {
			return errors.Errorf("bundle contains duplicate ordinal %d for document %q", chunk.Ordinal, chunk.DocumentID)
		}
		ordinals[chunk.DocumentID][chunk.Ordinal] = struct{}{}
		chunkIDs[chunk.ID] = struct{}{}
		documentIDs[chunk.DocumentID] = struct{}{}
	}
	// Structural chunkers may intentionally emit no chunks for admitted
	// documents that contain no indexable declarations. More stored document
	// identities than source documents is always invalid.
	if len(documentIDs) > documentCount {
		return errors.New("bundle chunk document count exceeds manifest corpus count")
	}
	return nil
}

func readJSON(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return errors.Wrap(err, "read JSON")
	}
	if err := jsonutil.DecodeStrictInto(data, target); err != nil {
		return errors.Wrap(err, "decode JSON")
	}
	return nil
}
