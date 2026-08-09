package representations

import (
	"context"
	"strings"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
)

func echoBatch(prefix string) BatchGenerate {
	return func(_ context.Context, requests []rag.GenerationRequest) ([]rag.GenerationResult, error) {
		results := make([]rag.GenerationResult, len(requests))
		for i := range requests {
			results[i] = rag.GenerationResult{Text: prefix}
		}
		return results, nil
	}
}

func labChunks() ([]rag.Document, []rag.Chunk) {
	text := "# Thuja Green Giant\n\nIntro paragraph.\n\n## Planting\n\n### Spacing\n\nPlant 5 feet apart."
	document := rag.Document{ID: "doc-1", Title: "Thuja Green Giant", Text: text}
	start := strings.Index(text, "Plant 5 feet apart.")
	chunk := rag.Chunk{
		ID: "chunk-1", DocumentID: "doc-1", Ordinal: 0,
		Range: rag.Range{ByteStart: start, ByteEnd: len(text)},
		Text:  "Plant 5 feet apart.", ContentDigest: "digest-1",
	}
	return []rag.Document{document}, []rag.Chunk{chunk}
}

func TestHeadingPathWalksToTheGoverningHeading(t *testing.T) {
	documents, chunks := labChunks()
	path := HeadingPath(documents[0], chunks[0])
	want := "Thuja Green Giant > Planting > Spacing"
	if path != want {
		t.Fatalf("heading path = %q, want %q", path, want)
	}
}

func TestHeadingPathResetsDeeperLevels(t *testing.T) {
	text := "## First\n\n### Deep\n\n## Second\n\nbody"
	document := rag.Document{ID: "d", Title: "T", Text: text}
	chunk := rag.Chunk{Range: rag.Range{ByteStart: strings.Index(text, "body")}}
	if path := HeadingPath(document, chunk); path != "T > Second" {
		t.Fatalf("heading path = %q, want %q (stale ### must not survive a new ##)", path, "T > Second")
	}
}

func TestContextualLiteCarriesBlurbAndChunkBody(t *testing.T) {
	documents, chunks := labChunks()
	var seenInput string
	generate := func(_ context.Context, requests []rag.GenerationRequest) ([]rag.GenerationResult, error) {
		seenInput = requests[0].Text
		return []rag.GenerationResult{{Text: "Spacing guidance for Thuja Green Giant hedges."}}, nil
	}
	reps, err := Contextual(context.Background(), documents, chunks, generate, ContextualSpec{
		Model: "glm", Prompt: "situate", LeadRunes: 40, IncludeChunkBody: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(reps) != 1 || reps[0].Kind != KindContextual {
		t.Fatalf("reps = %+v", reps)
	}
	if reps[0].Text != "Spacing guidance for Thuja Green Giant hedges.\nPlant 5 feet apart." {
		t.Fatalf("contextual text = %q", reps[0].Text)
	}
	if reps[0].ChunkID != "chunk-1" || reps[0].Model != "glm" || reps[0].PromptDigest == "" {
		t.Fatalf("provenance missing: %+v", reps[0])
	}
	for _, needle := range []string{"Thuja Green Giant", "Planting > Spacing", "Plant 5 feet apart."} {
		if !strings.Contains(seenInput, needle) {
			t.Fatalf("prompt input misses %q:\n%s", needle, seenInput)
		}
	}
}

func TestContextualBlurbOnlyOmitsChunkBody(t *testing.T) {
	documents, chunks := labChunks()
	reps, err := Contextual(context.Background(), documents, chunks, echoBatch("blurb"), ContextualSpec{
		Model: "glm", Prompt: "situate", IncludeChunkBody: false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if reps[0].Text != "blurb" {
		t.Fatalf("blurb-only text = %q", reps[0].Text)
	}
}

func TestGeneratedQuestionsSplitsAndStripsLines(t *testing.T) {
	_, chunks := labChunks()
	generate := echoBatch("- How far apart should I plant Thuja?\n2. What spacing for a hedge?\n\n```")
	reps, err := GeneratedQuestions(context.Background(), chunks, generate, GeneratedSpec{Model: "glm", Prompt: "ask"})
	if err != nil {
		t.Fatal(err)
	}
	if len(reps) != 2 {
		t.Fatalf("want 2 question reps, got %d: %+v", len(reps), reps)
	}
	if reps[0].Text != "How far apart should I plant Thuja?" || reps[1].Text != "What spacing for a hedge?" {
		t.Fatalf("questions = %q, %q", reps[0].Text, reps[1].Text)
	}
	if reps[0].Kind != KindQuestion || reps[0].ChunkID != "chunk-1" {
		t.Fatalf("lineage broken: %+v", reps[0])
	}
}

func TestEntityExpansionsAppendsOneLine(t *testing.T) {
	_, chunks := labChunks()
	generate := echoBatch("Thuja Green Giant\narborvitae, Thuja standishii × plicata")
	reps, err := EntityExpansions(context.Background(), chunks, generate, GeneratedSpec{Model: "glm", Prompt: "entities"})
	if err != nil {
		t.Fatal(err)
	}
	want := "Plant 5 feet apart.\nThuja Green Giant; arborvitae, Thuja standishii × plicata"
	if reps[0].Text != want {
		t.Fatalf("entity text = %q, want %q", reps[0].Text, want)
	}
	if reps[0].Kind != KindEntities {
		t.Fatalf("kind = %q", reps[0].Kind)
	}
}

func TestGeneratedSummariesSkipsEmptyResponses(t *testing.T) {
	_, chunks := labChunks()
	reps, err := GeneratedSummaries(context.Background(), chunks, echoBatch("  \n"), GeneratedSpec{Model: "glm", Prompt: "sum"})
	if err != nil {
		t.Fatal(err)
	}
	if len(reps) != 0 {
		t.Fatalf("empty responses must produce no representations, got %+v", reps)
	}
}

func TestGeneratedRepresentationIdentityIncludesModelAndPrompt(t *testing.T) {
	_, chunks := labChunks()
	base, err := generated(chunks[0], KindSummary, "same generated text", "model-a", "prompt-a")
	if err != nil {
		t.Fatal(err)
	}
	otherModel, err := generated(chunks[0], KindSummary, "same generated text", "model-b", "prompt-a")
	if err != nil {
		t.Fatal(err)
	}
	otherPrompt, err := generated(chunks[0], KindSummary, "same generated text", "model-a", "prompt-b")
	if err != nil {
		t.Fatal(err)
	}
	if base.ID == otherModel.ID || base.ID == otherPrompt.ID || otherModel.ID == otherPrompt.ID {
		t.Fatalf("generated IDs must distinguish provenance: %q %q %q", base.ID, otherModel.ID, otherPrompt.ID)
	}
	if base.Model != "model-a" || base.PromptDigest == "" {
		t.Fatalf("provenance missing from representation: %+v", base)
	}
}

func TestBuildersRefuseNilGenerate(t *testing.T) {
	documents, chunks := labChunks()
	if _, err := Contextual(context.Background(), documents, chunks, nil, ContextualSpec{}); err == nil {
		t.Fatal("Contextual accepted a nil generation function")
	}
	if _, err := GeneratedQuestions(context.Background(), chunks, nil, GeneratedSpec{}); err == nil {
		t.Fatal("GeneratedQuestions accepted a nil generation function")
	}
}
