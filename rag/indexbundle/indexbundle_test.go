package indexbundle

import (
	"context"
	"database/sql"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	blevelib "github.com/blevesearch/bleve/v2"
	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/go-go-golems/ragkit/rag/chunking"
	"github.com/go-go-golems/ragkit/rag/embedding"
	_ "github.com/mattn/go-sqlite3"
	"github.com/stretchr/testify/require"
)

func fixtureInput(t *testing.T, outputRoot string) BuildInput {
	t.Helper()
	documents := []rag.Document{{
		ID: "doc-1", Title: "Trees",
		Text: "# Oak\nOak trees have lobed leaves.",
	}}
	documents[0].ContentDigest = digest.Text(documents[0].Text)
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
		Path: first.Path, QueryEmbedder: input.QueryEmbedder, EmbeddingProvider: "hash",
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

func TestOpenReportsCompletedServingStages(t *testing.T) {
	input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
	built, err := Build(t.Context(), input)
	require.NoError(t, err)
	var stages []OpenStage
	bundle, err := Open(t.Context(), OpenOptions{
		Path: built.Path, QueryEmbedder: input.QueryEmbedder,
		EmbeddingProvider: "hash", EmbeddingModel: "hash-v1-d16",
		EmbeddingDimensions: 16,
		ObserveStage:        func(stage OpenStage) { stages = append(stages, stage) },
	})
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, bundle.Close()) })
	require.Equal(t, []OpenStage{
		OpenStageManifest,
		OpenStageChunks,
		OpenStageRepresentations,
		OpenStageBackendsVerified,
		OpenStageLexicalOpened,
		OpenStageVectorOpened,
		OpenStageReady,
	}, stages)
}

func TestBuildReportsCompletedStageBoundaries(t *testing.T) {
	input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
	var stages []BuildStage
	input.ObserveStage = func(stage BuildStage) {
		stages = append(stages, stage)
	}

	_, err := Build(t.Context(), input)
	require.NoError(t, err)
	require.Equal(t, []BuildStage{
		BuildStageInputValidated,
		BuildStageIdentityPlanned,
		BuildStageTemporaryCreated,
		BuildStagePayloadsWritten,
		BuildStageLexicalBuilt,
		BuildStageVectorBuilt,
		BuildStageManifestWritten,
		BuildStageBundlePublished,
		BuildStageResultMeasured,
	}, stages)

	stages = nil
	_, err = Build(t.Context(), input)
	require.NoError(t, err)
	require.Equal(t, []BuildStage{
		BuildStageInputValidated,
		BuildStageIdentityPlanned,
		BuildStageExistingVerified,
		BuildStageResultMeasured,
	}, stages)
}

func TestBuildRejectsReuseForDifferentCorpusPath(t *testing.T) {
	input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
	_, err := Build(t.Context(), input)
	require.NoError(t, err)
	input.CorpusPath = "replacement/corpus.json"
	_, err = Build(t.Context(), input)
	require.ErrorContains(t, err, "differs from requested")
}

func TestValidateStoredChunksRejectsInvalidOrdinals(t *testing.T) {
	base := rag.Chunk{ID: "a", DocumentID: "doc", Ordinal: 0, Text: "a", ContentDigest: digest.Text("a"), Range: rag.Range{ByteEnd: 1}}
	negative := base
	negative.Ordinal = -1
	require.ErrorContains(t, validateStoredChunks([]rag.Chunk{negative}, 1), "negative ordinal")
	duplicate := base
	duplicate.ID = "b"
	require.ErrorContains(t, validateStoredChunks([]rag.Chunk{base, duplicate}, 1), "duplicate ordinal")
}

func TestBuildRejectsReuseWhenStoredSemanticIdentityIsCorrupt(t *testing.T) {
	input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
	result, err := Build(t.Context(), input)
	require.NoError(t, err)
	manifestPath := filepath.Join(result.Path, manifestName)
	data, err := os.ReadFile(manifestPath)
	require.NoError(t, err)
	var manifest Manifest
	require.NoError(t, json.Unmarshal(data, &manifest))
	manifest.CorpusDigest = "corrupt"
	data, err = json.Marshal(manifest)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(manifestPath, data, 0o600))

	_, err = Build(t.Context(), input)
	require.ErrorContains(t, err, "existing bundle identity is invalid")
}

func TestOpenAllowsAdmittedDocumentWithoutChunks(t *testing.T) {
	input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
	input.Documents = append(input.Documents, rag.Document{
		ID: "doc-package-only", Title: "Package only", Text: "package trees",
		ContentDigest: digest.Text("package trees"),
	})
	result, err := Build(t.Context(), input)
	require.NoError(t, err)
	require.Equal(t, 2, result.Manifest.DocumentCount)

	bundle, err := Open(t.Context(), OpenOptions{
		Path: result.Path, QueryEmbedder: input.QueryEmbedder, EmbeddingProvider: "hash",
		EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 16,
	})
	require.NoError(t, err)
	require.NoError(t, bundle.Close())
}

func TestIdentityChangesForEverySemanticInput(t *testing.T) {
	base := fixtureInput(t, t.TempDir())
	id, _, _, _, _, err := CalculateID(
		base.Documents, base.Chunks, base.Representations, base.Chunker,
		BackendIdentity{Backend: "bleve-bm25", Version: 1, Channel: "bm25", TitleBoost: 2, BodyBoost: 1},
		base.Embedding,
	)
	require.NoError(t, err)

	changedChunker := base.Chunker
	changedChunker.MaximumRunes++
	other, _, _, _, _, err := CalculateID(
		base.Documents, base.Chunks, base.Representations, changedChunker,
		BackendIdentity{Backend: "bleve-bm25", Version: 1, Channel: "bm25", TitleBoost: 2, BodyBoost: 1},
		base.Embedding,
	)
	require.NoError(t, err)
	require.NotEqual(t, id, other)

	changedEmbedding := *base.Embedding
	changedEmbedding.Model = "different"
	other, _, _, _, _, err = CalculateID(
		base.Documents, base.Chunks, base.Representations, base.Chunker,
		BackendIdentity{Backend: "bleve-bm25", Version: 1, Channel: "bm25", TitleBoost: 2, BodyBoost: 1},
		&changedEmbedding,
	)
	require.NoError(t, err)
	require.NotEqual(t, id, other)

	changedRepresentations := append([]rag.Representation(nil), base.Representations...)
	changedRepresentations[0].Text += " changed"
	other, _, _, _, _, err = CalculateID(
		base.Documents, base.Chunks, changedRepresentations, base.Chunker,
		BackendIdentity{Backend: "bleve-bm25", Version: 1, Channel: "bm25", TitleBoost: 2, BodyBoost: 1},
		base.Embedding,
	)
	require.NoError(t, err)
	require.NotEqual(t, id, other)

	changedChunks := append([]rag.Chunk(nil), base.Chunks...)
	changedChunks[0].Text += " changed"
	changedChunks[0].ContentDigest = digest.Text(changedChunks[0].Text)
	changedChunks[0].Range.ByteEnd = changedChunks[0].Range.ByteStart + len(changedChunks[0].Text)
	other, _, _, _, _, err = CalculateID(
		base.Documents, changedChunks, base.Representations, base.Chunker,
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

func TestBuildRejectsInvalidCorpusBeforeCreatingBackends(t *testing.T) {
	tests := []struct {
		name   string
		want   string
		mutate func(*BuildInput)
	}{
		{
			name: "duplicate document ID", want: "duplicate document ID",
			mutate: func(input *BuildInput) {
				input.Documents = append(input.Documents, input.Documents[0])
			},
		},
		{
			name: "duplicate chunk ID", want: "duplicate chunk ID",
			mutate: func(input *BuildInput) {
				input.Chunks = append(input.Chunks, input.Chunks[0])
			},
		},
		{
			name: "unknown document", want: "references unknown document",
			mutate: func(input *BuildInput) {
				input.Chunks[0].DocumentID = "missing"
			},
		},
		{
			name: "invalid source range", want: "invalid byte range",
			mutate: func(input *BuildInput) {
				input.Chunks[0].Range.ByteEnd = len(input.Documents[0].Text) + 1
			},
		},
		{
			name: "source text mismatch", want: "text does not match its source range",
			mutate: func(input *BuildInput) {
				input.Chunks[0].Text = "different"
				input.Chunks[0].ContentDigest = digest.Text("different")
			},
		},
		{
			name: "chunk digest mismatch", want: "content digest mismatch",
			mutate: func(input *BuildInput) {
				input.Chunks[0].ContentDigest = digest.Text("different")
			},
		},
		{
			name: "raw representation mismatch", want: "raw representation",
			mutate: func(input *BuildInput) {
				input.Representations[0].Text = "different raw text"
				input.Representations[0].ContentDigest = digest.Text(input.Representations[0].Text)
			},
		},
		{
			name: "chunker identity mismatch", want: "bundle declares",
			mutate: func(input *BuildInput) {
				input.Chunker.Name = "other-chunker"
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), "indexes")
			input := fixtureInput(t, root)
			test.mutate(&input)
			_, err := Build(t.Context(), input)
			require.ErrorContains(t, err, test.want)
			_, statErr := os.Stat(root)
			require.ErrorIs(t, statErr, os.ErrNotExist, "corpus validation must run before backend setup")
		})
	}
}

func TestBuildRejectsInvalidVectorBackendIdentity(t *testing.T) {
	for name, mutate := range map[string]func(*VectorIdentity){
		"backend": func(identity *VectorIdentity) { identity.Backend = "hnsw" },
		"version": func(identity *VectorIdentity) { identity.Version = 2 },
		"channel": func(identity *VectorIdentity) { identity.Channel = "" },
	} {
		t.Run(name, func(t *testing.T) {
			input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
			mutate(input.Embedding)
			_, err := Build(t.Context(), input)
			require.ErrorContains(t, err, "sqlite-exact backend version 1")
		})
	}
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
		Path: result.Path, QueryEmbedder: input.QueryEmbedder, EmbeddingProvider: "other",
		EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 16,
	})
	require.ErrorContains(t, err, "differs from bundle provider")
	_, err = Open(t.Context(), OpenOptions{
		Path: result.Path, QueryEmbedder: input.QueryEmbedder, EmbeddingProvider: "hash",
		EmbeddingModel: "other", EmbeddingDimensions: 16,
	})
	require.ErrorContains(t, err, "differs from bundle model")
	_, err = Open(t.Context(), OpenOptions{
		Path: result.Path, QueryEmbedder: input.QueryEmbedder, EmbeddingProvider: "hash",
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
		Path: result.Path, QueryEmbedder: input.QueryEmbedder, EmbeddingProvider: "hash",
		EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 16,
	})
	require.ErrorContains(t, err, "counts differ")
}

func TestOpenValidatesChunkDigestWithoutRawRepresentations(t *testing.T) {
	input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
	input.Representations[0].Kind = "summary"
	input.Representations[0].Text = "generated summary"
	input.Representations[0].ContentDigest = digest.Text(input.Representations[0].Text)
	result, err := Build(t.Context(), input)
	require.NoError(t, err)

	chunksPath := filepath.Join(result.Path, chunksName)
	var chunks []rag.Chunk
	require.NoError(t, readJSON(chunksPath, &chunks))
	chunks[0].Text += " tampered"
	data, err := json.Marshal(chunks)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(chunksPath, data, 0o600))

	_, err = Open(t.Context(), OpenOptions{
		Path: result.Path, QueryEmbedder: input.QueryEmbedder, EmbeddingProvider: "hash",
		EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 16,
	})
	require.ErrorContains(t, err, "content digest mismatch")
}

func TestOpenRejectsInconsistentStoredChunkRange(t *testing.T) {
	input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
	result, err := Build(t.Context(), input)
	require.NoError(t, err)
	chunksPath := filepath.Join(result.Path, chunksName)
	var chunks []rag.Chunk
	require.NoError(t, readJSON(chunksPath, &chunks))
	chunks[0].Range.ByteEnd++
	data, err := json.Marshal(chunks)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(chunksPath, data, 0o600))

	_, err = Open(t.Context(), OpenOptions{
		Path: result.Path, QueryEmbedder: input.QueryEmbedder, EmbeddingProvider: "hash",
		EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 16,
	})
	require.ErrorContains(t, err, "invalid stored byte range")
}

func TestOpenRejectsPersistedBackendContentChanges(t *testing.T) {
	t.Run("lexical", func(t *testing.T) {
		input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
		result, err := Build(t.Context(), input)
		require.NoError(t, err)
		path := filepath.Join(result.Path, bleveName)
		index, err := blevelib.Open(path)
		require.NoError(t, err)
		require.NoError(t, index.Index(input.Representations[0].ID, map[string]any{
			"representation_id": input.Representations[0].ID,
			"chunk_id":          input.Chunks[0].ID,
			"document_id":       input.Documents[0].ID,
			"kind":              input.Representations[0].Kind,
			"title":             input.Documents[0].Title,
			"body":              "different indexed body",
		}))
		require.NoError(t, index.Close())

		_, err = Open(t.Context(), OpenOptions{
			Path: result.Path, QueryEmbedder: input.QueryEmbedder, EmbeddingProvider: "hash",
			EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 16,
		})
		require.ErrorContains(t, err, "lexical content differs")
	})

	t.Run("vector", func(t *testing.T) {
		input := fixtureInput(t, filepath.Join(t.TempDir(), "indexes"))
		result, err := Build(t.Context(), input)
		require.NoError(t, err)
		db, err := sql.Open("sqlite3", filepath.Join(result.Path, vectorName))
		require.NoError(t, err)
		_, err = db.Exec(`UPDATE embedding SET content_digest = 'different'`)
		require.NoError(t, err)
		require.NoError(t, db.Close())

		_, err = Open(t.Context(), OpenOptions{
			Path: result.Path, QueryEmbedder: input.QueryEmbedder, EmbeddingProvider: "hash",
			EmbeddingModel: "hash-v1-d16", EmbeddingDimensions: 16,
		})
		require.ErrorContains(t, err, "vector identity differs")
	})
}
