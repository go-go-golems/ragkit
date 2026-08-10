package geppetto

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/go-go-golems/geppetto/pkg/embeddings"
	"github.com/go-go-golems/ragkit/rag"
)

type embeddingProviderFixture struct {
	model   embeddings.EmbeddingModel
	vectors [][]float32
	err     error
	texts   []string
}

func (f *embeddingProviderFixture) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	vectors, err := f.GenerateBatchEmbeddings(ctx, []string{text})
	if err != nil || len(vectors) == 0 {
		return nil, err
	}
	return vectors[0], nil
}

func (f *embeddingProviderFixture) GenerateBatchEmbeddings(_ context.Context, texts []string) ([][]float32, error) {
	f.texts = append([]string(nil), texts...)
	return f.vectors, f.err
}

func (f *embeddingProviderFixture) GetModel() embeddings.EmbeddingModel { return f.model }

func TestNewEmbedderValidatesProviderMetadata(t *testing.T) {
	t.Parallel()

	var nilProvider embeddings.Provider
	for _, test := range []struct {
		name     string
		provider embeddings.Provider
	}{
		{name: "nil", provider: nilProvider},
		{name: "empty model", provider: &embeddingProviderFixture{model: embeddings.EmbeddingModel{Dimensions: 2}}},
		{name: "whitespace model", provider: &embeddingProviderFixture{model: embeddings.EmbeddingModel{Name: "  ", Dimensions: 2}}},
		{name: "zero dimensions", provider: &embeddingProviderFixture{model: embeddings.EmbeddingModel{Name: "embed"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if _, err := NewEmbedder(test.provider); err == nil {
				t.Fatal("expected constructor error")
			}
		})
	}
}

func TestEmbedderProjectsExactBatch(t *testing.T) {
	t.Parallel()

	provider := &embeddingProviderFixture{
		model: embeddings.EmbeddingModel{Name: "embed-v1", Dimensions: 2},
		vectors: [][]float32{
			{1, 0},
			{0, 1},
		},
	}
	adapter, err := NewEmbedder(provider)
	if err != nil {
		t.Fatal(err)
	}
	texts := []string{"first", "second"}
	result, err := adapter.Embed(t.Context(), rag.EmbeddingRequest{Model: "embed-v1", Texts: texts})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Vectors, provider.vectors) {
		t.Fatalf("vectors = %#v, want %#v", result.Vectors, provider.vectors)
	}
	if !reflect.DeepEqual(provider.texts, texts) {
		t.Fatalf("provider texts = %#v, want %#v", provider.texts, texts)
	}
	if !reflect.DeepEqual(texts, []string{"first", "second"}) {
		t.Fatalf("caller texts mutated: %#v", texts)
	}
}

func TestEmbedderRejectsInvalidRequestsBeforeProviderCall(t *testing.T) {
	t.Parallel()

	provider := &embeddingProviderFixture{
		model:   embeddings.EmbeddingModel{Name: "embed-v1", Dimensions: 2},
		vectors: [][]float32{{1, 0}},
	}
	adapter, err := NewEmbedder(provider)
	if err != nil {
		t.Fatal(err)
	}
	for _, request := range []rag.EmbeddingRequest{
		{Model: "different", Texts: []string{"text"}},
		{Model: "embed-v1"},
	} {
		provider.texts = nil
		if _, err := adapter.Embed(t.Context(), request); err == nil {
			t.Fatalf("expected request error for %#v", request)
		}
		if provider.texts != nil {
			t.Fatalf("provider called for invalid request %#v", request)
		}
	}
}

func TestEmbedderRejectsInvalidProviderResponses(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		vectors [][]float32
		want    string
	}{
		{name: "count mismatch", vectors: [][]float32{{1, 0}}, want: "count mismatch"},
		{name: "dimensions", vectors: [][]float32{{1}, {0, 1}}, want: "dimensions"},
		{name: "nan", vectors: [][]float32{{float32(math.NaN()), 0}, {0, 1}}, want: "finite"},
		{name: "positive infinity", vectors: [][]float32{{float32(math.Inf(1)), 0}, {0, 1}}, want: "finite"},
		{name: "negative infinity", vectors: [][]float32{{float32(math.Inf(-1)), 0}, {0, 1}}, want: "finite"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			provider := &embeddingProviderFixture{
				model:   embeddings.EmbeddingModel{Name: "embed-v1", Dimensions: 2},
				vectors: test.vectors,
			}
			adapter, err := NewEmbedder(provider)
			if err != nil {
				t.Fatal(err)
			}
			_, err = adapter.Embed(t.Context(), rag.EmbeddingRequest{
				Model: "embed-v1",
				Texts: []string{"first", "second"},
			})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want substring %q", err, test.want)
			}
		})
	}
}

func TestEmbedderWrapsProviderError(t *testing.T) {
	t.Parallel()

	providerErr := errors.New("fixture unavailable")
	provider := &embeddingProviderFixture{
		model: embeddings.EmbeddingModel{Name: "embed-v1", Dimensions: 2},
		err:   providerErr,
	}
	adapter, err := NewEmbedder(provider)
	if err != nil {
		t.Fatal(err)
	}
	_, err = adapter.Embed(t.Context(), rag.EmbeddingRequest{Texts: []string{"text"}})
	if !errors.Is(err, providerErr) {
		t.Fatalf("error = %v, want wrapped provider error", err)
	}
}

func TestNilEmbedderFailsClosed(t *testing.T) {
	t.Parallel()

	var adapter *Embedder
	if _, err := adapter.Embed(t.Context(), rag.EmbeddingRequest{Texts: []string{"text"}}); err == nil {
		t.Fatal("expected nil adapter error")
	}
}
