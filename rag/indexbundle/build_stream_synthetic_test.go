package indexbundle

import (
	"context"
	"flag"
	"fmt"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

var (
	syntheticDocuments  = flag.Int("synthetic-documents", 200, "documents in the staged bundle regression")
	syntheticChunks     = flag.Int("synthetic-chunks", 500, "chunks in the staged bundle regression")
	syntheticDimensions = flag.Int("synthetic-dimensions", 32, "vector dimensions in the staged bundle regression")
	syntheticBatchSize  = flag.Int("synthetic-batch-size", 64, "maximum admitted staging batch")
	syntheticMaxHeapMiB = flag.Uint64("synthetic-max-heap-mib", 128, "maximum heap-in-use at staged build boundaries")
)

// TestBuildStreamSyntheticShape is both a small normal regression and the
// configurable local capacity harness for the production corpus shape. Run the
// large form explicitly under /usr/bin/time -v; the test itself never requires
// a provider, database, or pre-existing cache.
func TestBuildStreamSyntheticShape(t *testing.T) {
	require.Positive(t, *syntheticDocuments)
	require.Positive(t, *syntheticChunks)
	require.Positive(t, *syntheticDimensions)
	require.Positive(t, *syntheticBatchSize)

	representations := *syntheticChunks * 2
	embedder := syntheticQueryEmbedder{dimensions: *syntheticDimensions}
	var maximumHeapInuse uint64
	input := StreamInput{
		OutputRoot: t.TempDir(), CorpusPath: "synthetic/corpus.json", BatchSize: *syntheticBatchSize,
		ScratchDirectory: filepath.Join(t.TempDir(), "tmp"),
		Chunker:          ChunkerIdentity{Name: "synthetic", MaximumRunes: 1024},
		Embedding: &VectorIdentity{
			Backend: "sqlite-exact", Version: 1, Channel: "vector", Provider: "synthetic",
			Model: "synthetic", Dimensions: *syntheticDimensions,
		},
		QueryEmbedder: embedder,
		Produce: func(ctx context.Context, stager *Stager) error {
			if err := produceSyntheticDocuments(ctx, stager, *syntheticDocuments, *syntheticBatchSize); err != nil {
				return err
			}
			if err := produceSyntheticChunks(ctx, stager, *syntheticDocuments, *syntheticChunks, *syntheticBatchSize); err != nil {
				return err
			}
			if err := produceSyntheticRepresentations(ctx, stager, *syntheticDocuments, *syntheticChunks, *syntheticBatchSize); err != nil {
				return err
			}
			return produceSyntheticVectors(ctx, stager, representations, *syntheticDimensions, *syntheticBatchSize)
		},
		ObserveStage: func(stage BuildStage) {
			var memory runtime.MemStats
			runtime.ReadMemStats(&memory)
			maximumHeapInuse = max(maximumHeapInuse, memory.HeapInuse)
			t.Logf("stage=%s heap_alloc=%d heap_inuse=%d heap_sys=%d sys=%d num_gc=%d",
				stage, memory.Alloc, memory.HeapInuse, memory.HeapSys, memory.Sys, memory.NumGC)
		},
	}

	result, err := BuildStream(t.Context(), input)
	require.NoError(t, err)
	require.Equal(t, *syntheticDocuments, result.Manifest.DocumentCount)
	require.Equal(t, *syntheticChunks, result.Manifest.ChunkCount)
	require.Equal(t, representations, result.Manifest.RepresentationCount)
	require.Positive(t, result.BleveBytes)
	require.Positive(t, result.VectorBytes)
	require.Less(t, maximumHeapInuse, *syntheticMaxHeapMiB*1024*1024)
}

func produceSyntheticDocuments(ctx context.Context, stager *Stager, count, batchSize int) error {
	return produceSyntheticBatches(count, batchSize, func(start, end int) error {
		batch := make([]rag.Document, 0, end-start)
		for index := start; index < end; index++ {
			text := fmt.Sprintf("synthetic document %06d", index)
			batch = append(batch, rag.Document{
				ID: fmt.Sprintf("doc-%06d", index), Title: fmt.Sprintf("Document %06d", index),
				Text: text, ContentDigest: digest.Text(text),
			})
		}
		return stager.AddDocuments(ctx, batch)
	})
}

func produceSyntheticChunks(ctx context.Context, stager *Stager, documentCount, count, batchSize int) error {
	return produceSyntheticBatches(count, batchSize, func(start, end int) error {
		batch := make([]rag.Chunk, 0, end-start)
		for index := start; index < end; index++ {
			documentIndex := index % documentCount
			text := fmt.Sprintf("synthetic document %06d", documentIndex)
			batch = append(batch, rag.Chunk{
				ID: fmt.Sprintf("chunk-%06d", index), DocumentID: fmt.Sprintf("doc-%06d", documentIndex),
				Ordinal: index / documentCount, Range: rag.Range{ByteEnd: len(text)},
				Text: text, ContentDigest: digest.Text(text), Chunker: "synthetic",
			})
		}
		return stager.AddChunks(ctx, batch)
	})
}

func produceSyntheticRepresentations(ctx context.Context, stager *Stager, documentCount, chunkCount, batchSize int) error {
	return produceSyntheticBatches(chunkCount*2, batchSize, func(start, end int) error {
		batch := make([]rag.Representation, 0, end-start)
		for index := start; index < end; index++ {
			chunkIndex := index / 2
			documentIndex := chunkIndex % documentCount
			text := fmt.Sprintf("synthetic document %06d", documentIndex)
			kind := "raw"
			if index%2 == 1 {
				kind = "breadcrumb"
			}
			batch = append(batch, rag.Representation{
				ID:      fmt.Sprintf("rep-%06d-%d", chunkIndex, index%2),
				ChunkID: fmt.Sprintf("chunk-%06d", chunkIndex), Kind: kind,
				Text: text, ContentDigest: digest.Text(text),
			})
		}
		return stager.AddRepresentations(ctx, batch)
	})
}

func produceSyntheticVectors(ctx context.Context, stager *Stager, count, dimensions, batchSize int) error {
	return produceSyntheticBatches(count, batchSize, func(start, end int) error {
		batch := make([]rag.Vector, 0, end-start)
		for index := start; index < end; index++ {
			values := make([]float32, dimensions)
			for dimension := range values {
				values[dimension] = float32((index+dimension)%101) / 101
			}
			batch = append(batch, rag.Vector{
				RepresentationID: fmt.Sprintf("rep-%06d-%d", index/2, index%2),
				Model:            "synthetic", Values: values,
			})
		}
		return stager.AddVectors(ctx, batch)
	})
}

func produceSyntheticBatches(count, batchSize int, produce func(start, end int) error) error {
	for start := 0; start < count; start += batchSize {
		if err := produce(start, min(start+batchSize, count)); err != nil {
			return err
		}
	}
	return nil
}

type syntheticQueryEmbedder struct{ dimensions int }

func (s syntheticQueryEmbedder) Embed(context.Context, rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	return rag.EmbeddingResult{Vectors: [][]float32{make([]float32, s.dimensions)}}, nil
}
