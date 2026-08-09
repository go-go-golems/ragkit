package representations

import (
	"context"
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
)

func TestRepresentationConveniencesRejectNilCollaborators(t *testing.T) {
	_, err := Summaries(t.Context(), nil, nil)
	if err == nil {
		t.Fatal("Summaries accepted a nil summarizer")
	}
	_, err = Questions(t.Context(), nil, nil)
	if err == nil {
		t.Fatal("Questions accepted a nil questioner")
	}
}

func chunk(id, text string) rag.Chunk {
	return rag.Chunk{
		ID: id, DocumentID: "doc", Text: text,
		ContentDigest: digest.Text(text), Chunker: "test",
	}
}

func TestSummaryRepresentationPointsToSourceChunk(t *testing.T) {
	t.Parallel()
	c := chunk("chunk-1", "Thuja Green Giant is a fast-growing arborvitae. It reaches 3 feet per year.")
	reps, err := Summaries(context.Background(), []rag.Chunk{c}, ExtractiveSummarizer{})
	if err != nil {
		t.Fatalf("Summaries error = %v", err)
	}
	if len(reps) != 1 {
		t.Fatalf("reps = %d, want 1", len(reps))
	}
	rep := reps[0]
	if rep.Kind != KindSummary {
		t.Fatalf("kind = %q, want %q", rep.Kind, KindSummary)
	}
	if rep.ChunkID != "chunk-1" {
		t.Fatalf("ChunkID = %q, want chunk-1", rep.ChunkID)
	}
	if rep.Text == c.Text {
		t.Fatalf("summary equals source text; extractive summarizer did not shorten")
	}
}

func TestHydrateMapsHitsToSourceChunks(t *testing.T) {
	t.Parallel()
	chunks := []rag.Chunk{chunk("a", "alpha"), chunk("b", "beta")}
	hits := []rag.Hit{
		{ChunkID: "b", RepresentationID: "rep-summary-b"},
		{ChunkID: "a", RepresentationID: "rep-summary-a"},
	}
	hydrated, err := Hydrate(hits, chunks)
	if err != nil {
		t.Fatalf("Hydrate error = %v", err)
	}
	if len(hydrated) != 2 {
		t.Fatalf("hydrated = %d, want 2", len(hydrated))
	}
	if hydrated[0].ID != "b" || hydrated[1].ID != "a" {
		t.Fatalf("hydration order = %s, %s; want b, a", hydrated[0].ID, hydrated[1].ID)
	}
}

func TestHydrateFailsOnUnknownChunk(t *testing.T) {
	t.Parallel()
	hits := []rag.Hit{{ChunkID: "missing"}}
	if _, err := Hydrate(hits, nil); err == nil {
		t.Fatalf("Hydrate should fail on an unknown chunk")
	}
}

func TestComposeDeduplicatesByID(t *testing.T) {
	t.Parallel()
	raw := []rag.Representation{{ID: "rep-1", Kind: "raw", ChunkID: "c"}}
	summary := []rag.Representation{
		{ID: "rep-2", Kind: KindSummary, ChunkID: "c"},
		{ID: "rep-1", Kind: "raw", ChunkID: "c"}, // duplicate of raw
	}
	combined := Compose(raw, summary)
	if len(combined) != 2 {
		t.Fatalf("combined = %d, want 2 after dedup", len(combined))
	}
}

func TestEnsureRawBackfillsMissingRaw(t *testing.T) {
	t.Parallel()
	chunks := []rag.Chunk{chunk("a", "alpha"), chunk("b", "beta")}
	// Only a summary for chunk a; raw for b is missing entirely.
	reps := []rag.Representation{
		{ID: "rep-s-a", Kind: KindSummary, ChunkID: "a", Text: "alpha summary"},
	}
	complete := EnsureRaw(chunks, reps)
	kinds := map[string]string{}
	for _, rep := range complete {
		kinds[rep.Kind+":"+rep.ChunkID] = rep.ID
	}
	if _, ok := kinds["raw:a"]; !ok {
		t.Fatalf("EnsureRaw did not backfill raw for chunk a")
	}
	if _, ok := kinds["raw:b"]; !ok {
		t.Fatalf("EnsureRaw did not backfill raw for chunk b")
	}
	if _, ok := kinds["summary:a"]; !ok {
		t.Fatalf("EnsureRaw dropped the existing summary")
	}
}

func TestExtractiveSummarizerCapsLongText(t *testing.T) {
	t.Parallel()
	long := "This is a very long sentence that goes on and on without any period for a long time " +
		"and should be capped by the max runes limit of the summarizer"
	c := chunk("chunk-1", long)
	summary, err := (ExtractiveSummarizer{MaxRunes: 40}).Summarize(context.Background(), c)
	if err != nil {
		t.Fatalf("Summarize error = %v", err)
	}
	if len([]rune(summary)) > 40 {
		t.Fatalf("summary length = %d runes, want <= 40", len([]rune(summary)))
	}
}

func TestExtractiveSummarizerDoesNotIncludeBoundaryPastMaxRunes(t *testing.T) {
	summary, err := (ExtractiveSummarizer{MaxRunes: 5}).Summarize(context.Background(), chunk("chunk", "abcde. later"))
	if err != nil {
		t.Fatal(err)
	}
	if got := len([]rune(summary)); got != 5 {
		t.Fatalf("summary length = %d, want 5: %q", got, summary)
	}
}

func TestExtractiveSummarizerOneRuneCapIncludesOnlyEllipsis(t *testing.T) {
	summary, err := (ExtractiveSummarizer{MaxRunes: 1}).Summarize(context.Background(), chunk("chunk", "long"))
	if err != nil {
		t.Fatal(err)
	}
	if summary != "…" {
		t.Fatalf("summary = %q, want ellipsis", summary)
	}
}

func TestExtractiveSummarizerRecognizesQuestionAndExclamationEndings(t *testing.T) {
	t.Parallel()
	for _, want := range []string{"Ready?", "Stop!"} {
		summary, err := (ExtractiveSummarizer{}).Summarize(context.Background(), chunk("chunk", want+" Later sentence."))
		if err != nil {
			t.Fatal(err)
		}
		if summary != want {
			t.Fatalf("summary = %q, want %q", summary, want)
		}
	}
}

func TestExtractiveSummarizerUsesRuneOffsetsForSentenceBoundaries(t *testing.T) {
	t.Parallel()
	text := "éééé? Later sentence."
	summary, err := (ExtractiveSummarizer{MaxRunes: 5}).Summarize(context.Background(), chunk("chunk", text))
	if err != nil {
		t.Fatal(err)
	}
	if summary != "éééé?" {
		t.Fatalf("summary = %q, want rune-safe first sentence", summary)
	}
}
