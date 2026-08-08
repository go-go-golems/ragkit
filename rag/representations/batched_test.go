package representations

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
)

func batchedFixture(chunksPerDoc ...int) ([]rag.Document, []rag.Chunk) {
	documents := make([]rag.Document, 0, len(chunksPerDoc))
	chunks := make([]rag.Chunk, 0)
	for d, count := range chunksPerDoc {
		var text strings.Builder
		text.WriteString(fmt.Sprintf("# Doc %d\n\n", d))
		docID := fmt.Sprintf("doc-%d", d)
		starts := make([]int, count)
		for c := 0; c < count; c++ {
			starts[c] = text.Len()
			text.WriteString(fmt.Sprintf("chunk %d of document %d. ", c, d))
		}
		document := rag.Document{ID: docID, Title: fmt.Sprintf("Doc %d", d), Text: text.String()}
		documents = append(documents, document)
		for c := 0; c < count; c++ {
			end := len(document.Text)
			if c+1 < count {
				end = starts[c+1]
			}
			chunks = append(chunks, rag.Chunk{
				ID: fmt.Sprintf("%s-chunk-%d", docID, c), DocumentID: docID, Ordinal: c,
				Range: rag.Range{ByteStart: starts[c], ByteEnd: end},
				Text:  document.Text[starts[c]:end], ContentDigest: "d",
			})
		}
	}
	return documents, chunks
}

func TestGroupByDocumentNeverMixesDocumentsAndCapsSize(t *testing.T) {
	documents, chunks := batchedFixture(5, 2)
	groups, err := groupByDocument(documents, chunks, 3)
	if err != nil {
		t.Fatal(err)
	}
	// doc-0: 5 chunks -> groups of 3+2; doc-1: 2 chunks -> one group.
	if len(groups) != 3 {
		t.Fatalf("groups = %d, want 3", len(groups))
	}
	if len(groups[0].chunks) != 3 || len(groups[1].chunks) != 2 || len(groups[2].chunks) != 2 {
		t.Fatalf("group sizes = %d/%d/%d", len(groups[0].chunks), len(groups[1].chunks), len(groups[2].chunks))
	}
	if groups[2].document.ID != "doc-1" {
		t.Fatalf("third group document = %s", groups[2].document.ID)
	}
}

func TestParseBatchedResponseToleratesFencesAndProse(t *testing.T) {
	response := "Here you go:\n```json\n[{\"chunk\": 1, \"text\": \"first\"}, {\"chunk\": 2, \"text\": \"second\"}]\n```"
	texts := parseBatchedResponse(response, 2)
	if texts[1] != "first" || texts[2] != "second" {
		t.Fatalf("texts = %v", texts)
	}
}

func TestParseBatchedResponseDropsOutOfRangeAndDuplicates(t *testing.T) {
	response := `[{"chunk": 0, "text": "x"}, {"chunk": 1, "text": "a"}, {"chunk": 1, "text": "b"}, {"chunk": 9, "text": "y"}, {"chunk": 2, "text": ""}]`
	texts := parseBatchedResponse(response, 2)
	if len(texts) != 1 || texts[1] != "a" {
		t.Fatalf("texts = %v", texts)
	}
}

func TestGenerateBatchedRepairsMissingChunksWithThePerChunkPrompt(t *testing.T) {
	documents, chunks := batchedFixture(3)
	var prompts []string
	generate := func(_ context.Context, requests []rag.GenerationRequest) ([]rag.GenerationResult, error) {
		results := make([]rag.GenerationResult, len(requests))
		for i, request := range requests {
			prompts = append(prompts, request.Prompt)
			if request.Prompt == "BATCHED" {
				// Answer only chunks 1 and 3; chunk 2 must be repaired.
				results[i] = rag.GenerationResult{Text: `[{"chunk":1,"text":"s1"},{"chunk":3,"text":"s3"}]`}
			} else {
				results[i] = rag.GenerationResult{Text: "repaired"}
			}
		}
		return results, nil
	}
	texts, err := generateBatched(context.Background(), documents, chunks, generate, KindSummary,
		BatchedSpec{Model: "glm", Prompt: "BATCHED", FallbackPrompt: "PER-CHUNK"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if texts[chunks[0].ID].text != "s1" || texts[chunks[2].ID].text != "s3" {
		t.Fatalf("batched texts missing: %v", texts)
	}
	if texts[chunks[1].ID].text != "repaired" {
		t.Fatalf("chunk 2 not repaired: %v", texts)
	}
	if prompts[len(prompts)-1] != "PER-CHUNK" {
		t.Fatalf("repair must use the per-chunk prompt, got %q", prompts[len(prompts)-1])
	}
}

func TestGenerateBatchedRejectsWrongRepairResultCount(t *testing.T) {
	documents, chunks := batchedFixture(2)
	generate := func(_ context.Context, requests []rag.GenerationRequest) ([]rag.GenerationResult, error) {
		if requests[0].Prompt == "BATCHED" {
			return []rag.GenerationResult{{Text: `[]`}}, nil
		}
		return []rag.GenerationResult{{Text: "only one"}}, nil
	}
	_, err := generateBatched(context.Background(), documents, chunks, generate, KindSummary,
		BatchedSpec{Model: "glm", Prompt: "BATCHED", FallbackPrompt: "PER-CHUNK"}, false)
	if err == nil || !strings.Contains(err.Error(), "repair returned 1 results for 2 chunks") {
		t.Fatalf("error = %v", err)
	}
}

func TestRepairedRepresentationUsesFallbackPromptDigest(t *testing.T) {
	documents, chunks := batchedFixture(2)
	generate := func(_ context.Context, requests []rag.GenerationRequest) ([]rag.GenerationResult, error) {
		if requests[0].Prompt == "BATCHED" {
			return []rag.GenerationResult{{Text: `[{"chunk":1,"text":"batched"}]`}}, nil
		}
		return []rag.GenerationResult{{Text: "repaired"}}, nil
	}
	reps, err := GeneratedSummariesBatched(context.Background(), documents, chunks, generate,
		BatchedSpec{Model: "glm", Prompt: "BATCHED", FallbackPrompt: "PER-CHUNK"})
	if err != nil {
		t.Fatal(err)
	}
	if reps[0].PromptDigest != digest.Text("BATCHED") {
		t.Fatalf("batched prompt digest = %q", reps[0].PromptDigest)
	}
	if reps[1].PromptDigest != digest.Text("PER-CHUNK") {
		t.Fatalf("repair prompt digest = %q", reps[1].PromptDigest)
	}
}

func TestContextualBatchedAssemblesBlurbAndBody(t *testing.T) {
	documents, chunks := batchedFixture(2)
	generate := func(_ context.Context, requests []rag.GenerationRequest) ([]rag.GenerationResult, error) {
		if !strings.Contains(requests[0].Text, "Document lead:") {
			t.Fatalf("contextual group input must carry the document lead:\n%s", requests[0].Text)
		}
		return []rag.GenerationResult{{Text: `[{"chunk":1,"text":"b1"},{"chunk":2,"text":"b2"}]`}}, nil
	}
	reps, err := ContextualBatched(context.Background(), documents, chunks, generate,
		BatchedSpec{Model: "glm", Prompt: "P", FallbackPrompt: "F"}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(reps) != 2 || !strings.HasPrefix(reps[0].Text, "b1\n") {
		t.Fatalf("reps = %+v", reps)
	}
	if reps[0].Kind != KindContextual || reps[0].ChunkID != chunks[0].ID {
		t.Fatalf("lineage broken: %+v", reps[0])
	}
}

// statementsGenerate answers the batched statement prompt for chunks 1 and 3
// (fenced, bulleted, and numbered lines included) and repairs chunk 2 with
// the per-chunk fallback prompt.
func statementsGenerate(repairPrompts *[]string) BatchGenerate {
	return func(_ context.Context, requests []rag.GenerationRequest) ([]rag.GenerationResult, error) {
		results := make([]rag.GenerationResult, len(requests))
		for i, request := range requests {
			if request.Prompt == "BATCHED" {
				results[i] = rag.GenerationResult{
					Text: "```json\n[{\"chunk\":1,\"text\":\"- fact a\\nfact b\"},{\"chunk\":3,\"text\":\"fact c\"}]\n```",
				}
			} else {
				if repairPrompts != nil {
					*repairPrompts = append(*repairPrompts, request.Prompt)
				}
				results[i] = rag.GenerationResult{Text: "1. repaired fact\n2. second repaired fact"}
			}
		}
		return results, nil
	}
}

func TestGeneratedStatementsBatchedSplitsLinesAndRepairsMissingChunks(t *testing.T) {
	documents, chunks := batchedFixture(3)
	var repairPrompts []string
	reps, err := GeneratedStatementsBatched(context.Background(), documents, chunks, statementsGenerate(&repairPrompts),
		BatchedSpec{Model: "glm", Prompt: "BATCHED", FallbackPrompt: "PER-CHUNK"})
	if err != nil {
		t.Fatal(err)
	}
	// chunk 1: two statements; chunk 2 (repaired): two; chunk 3: one.
	want := []struct{ chunkID, text string }{
		{chunks[0].ID, "fact a"},
		{chunks[0].ID, "fact b"},
		{chunks[1].ID, "repaired fact"},
		{chunks[1].ID, "second repaired fact"},
		{chunks[2].ID, "fact c"},
	}
	if len(reps) != len(want) {
		t.Fatalf("want %d statement reps, got %d: %+v", len(want), len(reps), reps)
	}
	for i, rep := range reps {
		if rep.ChunkID != want[i].chunkID || rep.Text != want[i].text {
			t.Fatalf("rep %d = (%q, %q), want (%q, %q)", i, rep.ChunkID, rep.Text, want[i].chunkID, want[i].text)
		}
		if rep.Kind != KindStatement {
			t.Fatalf("rep %d kind = %q, want %q", i, rep.Kind, KindStatement)
		}
	}
	if len(repairPrompts) != 1 || repairPrompts[0] != "PER-CHUNK" {
		t.Fatalf("repair must use the per-chunk fallback prompt exactly once, got %v", repairPrompts)
	}
}

func TestGeneratedStatementsBatchedIsDeterministicAcrossRuns(t *testing.T) {
	documents, chunks := batchedFixture(3)
	spec := BatchedSpec{Model: "glm", Prompt: "BATCHED", FallbackPrompt: "PER-CHUNK"}
	first, err := GeneratedStatementsBatched(context.Background(), documents, chunks, statementsGenerate(nil), spec)
	if err != nil {
		t.Fatal(err)
	}
	second, err := GeneratedStatementsBatched(context.Background(), documents, chunks, statementsGenerate(nil), spec)
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

func TestBatchedCallCeilingCountsGroupsPlusRepairs(t *testing.T) {
	if got := BatchedCallCeiling(25, 12); got != 3+25 {
		t.Fatalf("ceiling = %d, want 28", got)
	}
	if got := BatchedCallCeiling(0, 12); got != 0 {
		t.Fatalf("ceiling for zero chunks = %d", got)
	}
}
