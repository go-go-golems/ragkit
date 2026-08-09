package indexbundle

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
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
	return manifest, nil
}

func Open(ctx context.Context, options OpenOptions) (*Bundle, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	manifest, err := LoadManifest(options.Path)
	if err != nil {
		return nil, errors.Wrap(err, "load index bundle manifest")
	}
	data, err := loadData(options.Path, manifest)
	if err != nil {
		return nil, err
	}
	if err := validateStoredIdentity(manifest, data); err != nil {
		return nil, err
	}
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
		Manifest: manifest, Chunks: data.chunks,
		Representations: data.representations,
		Lexical:         lexical, Vector: vector,
	}, nil
}

func validateStoredIdentity(manifest Manifest, data bundleData) error {
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
		return errors.New("bundle manifest identity does not match stored data")
	}
	if manifest.Vector != nil {
		if manifest.Vector.RepresentationDigest != representationDigest {
			return errors.New("bundle vector representation digest mismatch")
		}
	}
	return nil
}

type bundleData struct {
	chunks          []rag.Chunk
	representations []rag.Representation
}

func loadData(path string, manifest Manifest) (bundleData, error) {
	var chunks []rag.Chunk
	if err := readJSON(filepath.Join(path, chunksName), &chunks); err != nil {
		return bundleData{}, errors.Wrap(err, "load bundle chunks")
	}
	var representations []rag.Representation
	if err := readJSON(filepath.Join(path, representationsName), &representations); err != nil {
		return bundleData{}, errors.Wrap(err, "load bundle representations")
	}
	if len(chunks) != manifest.ChunkCount ||
		len(representations) != manifest.RepresentationCount {
		return bundleData{}, errors.New("bundle data counts differ from manifest")
	}
	if err := validateStoredChunks(chunks, manifest.DocumentCount); err != nil {
		return bundleData{}, err
	}
	if err := rag.ValidateRepresentations(chunks, representations); err != nil {
		return bundleData{}, errors.Wrap(err, "validate bundle representations")
	}
	bleveData, err := os.ReadFile(filepath.Join(path, bleveName, "rag-manifest.json"))
	if err != nil {
		return bundleData{}, errors.Wrap(err, "read bundle lexical manifest")
	}
	var lexicalManifest bleveindex.Manifest
	if err := json.Unmarshal(bleveData, &lexicalManifest); err != nil {
		return bundleData{}, errors.Wrap(err, "decode bundle lexical manifest")
	}
	if lexicalIdentity(lexicalManifest) != manifest.Lexical ||
		lexicalManifest.RepresentationCount != manifest.RepresentationCount {
		return bundleData{}, errors.New("bundle lexical identity differs from manifest")
	}
	lexicalDigest, err := bleveindex.InspectContentDigest(filepath.Join(path, bleveName))
	if err != nil {
		return bundleData{}, errors.Wrap(err, "inspect bundle lexical content")
	}
	if lexicalDigest != manifest.Lexical.ContentDigest {
		return bundleData{}, errors.New("bundle lexical content differs from manifest")
	}
	if manifest.Vector != nil {
		vectorManifest, err := sqliteexact.Inspect(filepath.Join(path, vectorName))
		if err != nil {
			return bundleData{}, errors.Wrap(err, "inspect bundle vector index")
		}
		if vectorManifest.Model != manifest.Vector.Model ||
			vectorManifest.Dimensions != manifest.Vector.Dimensions ||
			vectorManifest.RepresentationCount != manifest.RepresentationCount ||
			vectorManifest.ContentDigest != manifest.Vector.ContentDigest {
			return bundleData{}, errors.New("bundle vector identity differs from manifest")
		}
	}
	// Corpus identity cannot be reconstructed from chunks because overlap and
	// Markdown boundaries duplicate text. Persist the digest check through the
	// manifest and bundle ID; callers validate source corpus when building.
	return bundleData{chunks: chunks, representations: representations}, nil
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
	if err := json.Unmarshal(data, target); err != nil {
		return errors.Wrap(err, "decode JSON")
	}
	return nil
}
