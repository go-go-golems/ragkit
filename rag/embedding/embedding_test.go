package embedding

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
)

func TestHashEmbedderDeterministic(t *testing.T) {
	t.Parallel()
	embedder := &HashEmbedder{Dimensions: 16}
	request := rag.EmbeddingRequest{Texts: []string{"oak tree", "oak tree"}}
	result, err := embedder.Embed(context.Background(), request)
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	for index := range result.Vectors[0] {
		if result.Vectors[0][index] != result.Vectors[1][index] {
			t.Fatal("equal inputs produced different vectors")
		}
	}
	const wantNonZero = float32(0.70710677)
	for index, value := range result.Vectors[0] {
		want := float32(0)
		if index == 11 {
			want = -wantNonZero
		}
		if index == 14 {
			want = wantNonZero
		}
		if math.Abs(float64(value-want)) > 1e-7 {
			t.Fatalf("vector[%d] = %.9f, want %.9f", index, value, want)
		}
	}
}

func TestRepresentationsBatches(t *testing.T) {
	t.Parallel()
	reps := []rag.Representation{{ID: "1", Text: "oak"}, {ID: "2", Text: "pine"}, {ID: "3", Text: "maple"}}
	vectors, _, err := Representations(context.Background(), &HashEmbedder{Dimensions: 8}, "hash", reps, 2)
	if err != nil {
		t.Fatalf("Representations() error = %v", err)
	}
	if len(vectors) != 3 {
		t.Fatalf("vectors = %d, want 3", len(vectors))
	}
}

func TestRepresentationsAggregatesReportedUsage(t *testing.T) {
	t.Parallel()
	embedder := &usageEmbedder{}
	reps := []rag.Representation{{ID: "1", Text: "oak"}, {ID: "2", Text: "pine"}}
	_, usage, err := Representations(context.Background(), embedder, "model", reps, 1)
	if err != nil {
		t.Fatalf("Representations() error = %v", err)
	}
	if usage.EmbeddingTokens == nil || *usage.EmbeddingTokens != 6 {
		t.Fatalf("EmbeddingTokens = %v, want 6", usage.EmbeddingTokens)
	}
	if usage.CostUSD == nil || *usage.CostUSD != 0.03 {
		t.Fatalf("CostUSD = %v, want 0.03", usage.CostUSD)
	}
	if usage.InputTokens != nil || usage.OutputTokens != nil {
		t.Fatalf("unreported token usage = %+v, want nil fields", usage)
	}
}

func TestRepresentationsPreservesMissingUsage(t *testing.T) {
	t.Parallel()
	reps := []rag.Representation{{ID: "1", Text: "oak"}}
	_, usage, err := Representations(context.Background(), missingUsageEmbedder{}, "model", reps, 1)
	if err != nil {
		t.Fatalf("Representations() error = %v", err)
	}
	if usage != (rag.Usage{}) {
		t.Fatalf("usage = %+v, want all fields nil", usage)
	}
}

func TestRepresentationsReturnsAccumulatedUsageOnLaterBatchFailure(t *testing.T) {
	t.Parallel()
	used := int64(7)
	embedder := &scriptedEmbedder{results: []scriptedEmbeddingResult{
		{result: rag.EmbeddingResult{Vectors: [][]float32{{1, 0}}, Usage: rag.Usage{EmbeddingTokens: &used}}},
		{err: errors.New("provider unavailable")},
	}}
	reps := []rag.Representation{{ID: "1", Text: "oak"}, {ID: "2", Text: "pine"}}
	vectors, usage, err := Representations(context.Background(), embedder, "model", reps, 1)
	if err == nil || !strings.Contains(err.Error(), "provider unavailable") {
		t.Fatalf("Representations() error = %v, want provider failure", err)
	}
	if vectors != nil {
		t.Fatalf("vectors = %+v, want nil on incomplete result", vectors)
	}
	if usage.EmbeddingTokens == nil || *usage.EmbeddingTokens != 7 {
		t.Fatalf("EmbeddingTokens = %v, want accumulated 7", usage.EmbeddingTokens)
	}
}

func TestRepresentationsRejectsDimensionChangesAcrossBatches(t *testing.T) {
	t.Parallel()
	embedder := &scriptedEmbedder{results: []scriptedEmbeddingResult{
		{result: rag.EmbeddingResult{Vectors: [][]float32{{1, 0}}}},
		{result: rag.EmbeddingResult{Vectors: [][]float32{{1, 0, 0}}}},
	}}
	reps := []rag.Representation{{ID: "1", Text: "oak"}, {ID: "2", Text: "pine"}}
	_, _, err := Representations(context.Background(), embedder, "model", reps, 1)
	if err == nil || !strings.Contains(err.Error(), "got 3, want 2") {
		t.Fatalf("Representations() error = %v, want cross-batch dimensions error", err)
	}
}

func TestRepresentationsRejectsNonFiniteVectors(t *testing.T) {
	t.Parallel()
	embedder := &scriptedEmbedder{results: []scriptedEmbeddingResult{{
		result: rag.EmbeddingResult{Vectors: [][]float32{{float32(math.NaN())}}},
	}}}
	_, _, err := Representations(context.Background(), embedder, "model", []rag.Representation{{ID: "1", Text: "oak"}}, 1)
	if err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("Representations() error = %v, want non-finite vector error", err)
	}
}

type scriptedEmbeddingResult struct {
	result rag.EmbeddingResult
	err    error
}

type scriptedEmbedder struct {
	results []scriptedEmbeddingResult
	calls   int
}

func (embedder *scriptedEmbedder) Embed(_ context.Context, _ rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	result := embedder.results[embedder.calls]
	embedder.calls++
	return result.result, result.err
}

type missingUsageEmbedder struct{}

func (missingUsageEmbedder) Embed(_ context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	vectors := make([][]float32, len(request.Texts))
	for index := range vectors {
		vectors[index] = []float32{1}
	}
	return rag.EmbeddingResult{Vectors: vectors}, nil
}

type usageEmbedder struct {
	calls int64
}

func (embedder *usageEmbedder) Embed(_ context.Context, request rag.EmbeddingRequest) (rag.EmbeddingResult, error) {
	embedder.calls++
	tokens := embedder.calls * 2
	cost := float64(embedder.calls) / 100
	return rag.EmbeddingResult{
		Vectors: [][]float32{{1}},
		Usage:   rag.Usage{EmbeddingTokens: &tokens, CostUSD: &cost},
	}, nil
}
