package representations

import (
	"context"
	"fmt"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
)

func documentSummaryGenerate(t *testing.T, documents []rag.Document) BatchGenerate {
	t.Helper()
	return func(_ context.Context, requests []rag.GenerationRequest) ([]rag.GenerationResult, error) {
		if len(requests) != len(documents) {
			t.Fatalf("want one request per document, got %d for %d documents", len(requests), len(documents))
		}
		results := make([]rag.GenerationResult, len(requests))
		for i, request := range requests {
			if request.Kind != KindDocumentSummary {
				t.Fatalf("request kind = %q, want %q", request.Kind, KindDocumentSummary)
			}
			if request.Text != documents[i].Text {
				t.Fatalf("request %d must carry the full document text, got %q", i, request.Text)
			}
			results[i] = rag.GenerationResult{Text: fmt.Sprintf("summary of %s", documents[i].Title)}
		}
		return results, nil
	}
}

func TestDocumentSummariesBuildOnePerDocumentHydratingToTheFirstChunk(t *testing.T) {
	documents, chunks := batchedFixture(3, 2)
	reps, err := DocumentSummaries(context.Background(), documents, chunks,
		documentSummaryGenerate(t, documents), GeneratedSpec{Model: "glm", Prompt: "doc-sum"})
	if err != nil {
		t.Fatal(err)
	}
	if len(reps) != 2 {
		t.Fatalf("want one representation per document, got %d: %+v", len(reps), reps)
	}
	// doc-0's first chunk is chunks[0], doc-1's first chunk is chunks[3].
	if reps[0].ChunkID != chunks[0].ID || reps[1].ChunkID != chunks[3].ID {
		t.Fatalf("document summaries must hydrate to the first chunk, got %q and %q", reps[0].ChunkID, reps[1].ChunkID)
	}
	if reps[0].Kind != KindDocumentSummary || reps[0].Text != "summary of Doc 0" {
		t.Fatalf("rep = %+v", reps[0])
	}
	if reps[0].Model != "glm" || reps[0].PromptDigest == "" {
		t.Fatalf("provenance missing: %+v", reps[0])
	}
}

func TestDocumentSummariesAreDeterministicAcrossRuns(t *testing.T) {
	documents, chunks := batchedFixture(2, 3)
	spec := GeneratedSpec{Model: "glm", Prompt: "doc-sum"}
	first, err := DocumentSummaries(context.Background(), documents, chunks, documentSummaryGenerate(t, documents), spec)
	if err != nil {
		t.Fatal(err)
	}
	second, err := DocumentSummaries(context.Background(), documents, chunks, documentSummaryGenerate(t, documents), spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != len(second) {
		t.Fatalf("run sizes differ: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i].ID != second[i].ID {
			t.Fatalf("representation %d id changed across runs: %q vs %q", i, first[i].ID, second[i].ID)
		}
	}
}

func TestDocumentSummariesRequireAChunkToHydrateTo(t *testing.T) {
	documents, _ := batchedFixture(1)
	called := false
	_, err := DocumentSummaries(context.Background(), documents, nil,
		func(context.Context, []rag.GenerationRequest) ([]rag.GenerationResult, error) {
			called = true
			return nil, nil
		}, GeneratedSpec{Model: "glm", Prompt: "doc-sum"})
	if err == nil {
		t.Fatal("a document without chunks must fail, not produce inadmissible evidence")
	}
	if called {
		t.Fatal("generation was called before hydration preflight completed")
	}
}

func TestDocumentSummariesRefuseNilGenerate(t *testing.T) {
	documents, chunks := batchedFixture(1)
	if _, err := DocumentSummaries(context.Background(), documents, chunks, nil, GeneratedSpec{}); err == nil {
		t.Fatal("DocumentSummaries accepted a nil generation function")
	}
}
