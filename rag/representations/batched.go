package representations

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// The batched engine sends one gateway call per group of chunks instead of
// one per chunk. The gateway's ~75 s latency is per-request, not per-token,
// so a 12-chunk group costs roughly one chunk's wall time — a ~12x speedup.
// The model answers with a JSON array ([{"chunk": N, "text": "..."}]); any
// chunk the response misses is repaired with a per-chunk call using the
// per-chunk canonical prompt, which makes repairs cache hits whenever a
// per-chunk run already generated that chunk.

// DefaultBatchGroupSize bounds how many chunks share one call. Twelve keeps
// the response (12 entries of ~100 tokens plus reasoning) far under the
// profile's response-token cap.
const DefaultBatchGroupSize = 12

// BatchedSpec configures one batched generation family.
type BatchedSpec struct {
	Model string
	// Prompt is the batched instruction (a Prompt*Batched constant).
	Prompt string
	// FallbackPrompt is the per-chunk instruction used to repair chunks the
	// batched response missed (the per-chunk canonical constant).
	FallbackPrompt string
	// GroupSize caps chunks per call; zero means DefaultBatchGroupSize.
	GroupSize int
}

func (s BatchedSpec) groupSize() int {
	if s.GroupSize > 0 {
		return s.GroupSize
	}
	return DefaultBatchGroupSize
}

// BatchedCallCeiling is the worst-case call count for one batched spec over
// the given chunk count: one possible document-bounded group plus one repair
// call per chunk. Chunk count alone cannot reveal document boundaries, so the
// conservative ceiling assumes every chunk belongs to a different document.
func BatchedCallCeiling(chunks, groupSize int) int {
	_ = groupSize // retained because callers state the configured batch size
	if chunks <= 0 {
		return 0
	}
	return 2 * chunks
}

// batchedGroup is one call's worth of chunks, all from one document.
type batchedGroup struct {
	document rag.Document
	chunks   []rag.Chunk
}

// groupByDocument splits the chunk list into per-document groups of at most
// size chunks, preserving order. Chunks never mix documents within a group:
// the prompt states one document's title, and a group crossing documents
// would attribute chunks to the wrong one.
func groupByDocument(documents []rag.Document, chunks []rag.Chunk, size int) ([]batchedGroup, error) {
	byID := make(map[string]rag.Document, len(documents))
	for _, document := range documents {
		byID[document.ID] = document
	}
	groups := make([]batchedGroup, 0, len(chunks)/size+1)
	var current *batchedGroup
	for _, chunk := range chunks {
		document, ok := byID[chunk.DocumentID]
		if !ok {
			return nil, errors.Errorf("chunk %s references unknown document %q", chunk.ID, chunk.DocumentID)
		}
		if current == nil || current.document.ID != chunk.DocumentID || len(current.chunks) >= size {
			groups = append(groups, batchedGroup{document: document})
			current = &groups[len(groups)-1]
		}
		current.chunks = append(current.chunks, chunk)
	}
	return groups, nil
}

// renderGroupInput renders one group's prompt input: the document title, the
// document lead when asked for (the contextual family), then the numbered
// chunks, each with its section path.
func renderGroupInput(group batchedGroup, withLead bool, leadRunesCount int) string {
	var input strings.Builder
	input.WriteString("Document title: ")
	input.WriteString(group.document.Title)
	input.WriteString("\n")
	if withLead {
		input.WriteString("Document lead:\n")
		input.WriteString(leadRunes(group.document.Text, leadRunesCount))
		input.WriteString("\n")
	}
	for index, chunk := range group.chunks {
		fmt.Fprintf(&input, "\n[%d] Section: %s\n", index+1, HeadingPath(group.document, chunk))
		input.WriteString(chunk.Text)
		input.WriteString("\n")
	}
	return input.String()
}

// batchedEntry is one element of the model's JSON answer.
type batchedEntry struct {
	Chunk int    `json:"chunk"`
	Text  string `json:"text"`
}

type generatedText struct {
	text   string
	prompt string
}

// parseBatchedResponse extracts the JSON array from a response, tolerating
// code fences and prose around it. Entries with out-of-range numbers or empty
// text are dropped; the caller repairs whatever is missing.
func parseBatchedResponse(response string, groupLen int) map[int]string {
	trimmed := strings.TrimSpace(response)
	texts := map[int]string{}
	for start := strings.IndexByte(trimmed, '['); start >= 0; {
		var entries []batchedEntry
		decoder := json.NewDecoder(strings.NewReader(trimmed[start:]))
		if err := decoder.Decode(&entries); err == nil {
			for _, entry := range entries {
				text := strings.TrimSpace(entry.Text)
				if entry.Chunk < 1 || entry.Chunk > groupLen || text == "" {
					continue
				}
				if _, seen := texts[entry.Chunk]; seen {
					continue
				}
				texts[entry.Chunk] = text
			}
			if len(texts) > 0 {
				return texts
			}
		}
		next := strings.IndexByte(trimmed[start+1:], '[')
		if next < 0 {
			break
		}
		start += next + 1
	}
	return texts
}

// generateBatched produces one text per chunk through grouped calls plus
// per-chunk repairs. contextual selects the contextual input format for both
// the group rendering and the repair calls. The returned map is keyed by
// chunk ID; a chunk whose repair also produced nothing is absent, and the
// assemblers skip it the same way they skip empty per-chunk responses.
func generateBatched(
	ctx context.Context,
	documents []rag.Document,
	chunks []rag.Chunk,
	generate BatchGenerate,
	kind string,
	spec BatchedSpec,
	contextual bool,
) (map[string]generatedText, error) {
	if generate == nil {
		return nil, errors.Errorf("the batched %s builder needs a generation function", kind)
	}
	groups, err := groupByDocument(documents, chunks, spec.groupSize())
	if err != nil {
		return nil, err
	}
	requests := make([]rag.GenerationRequest, len(groups))
	for i, group := range groups {
		requests[i] = rag.GenerationRequest{
			Kind: kind, Model: spec.Model, Prompt: spec.Prompt,
			Text: renderGroupInput(group, contextual, 500),
		}
	}
	results, err := generate(ctx, requests)
	if err != nil {
		return nil, errors.Wrapf(err, "generate batched %s representations", kind)
	}
	if len(results) != len(groups) {
		return nil, errors.Errorf("batched %s generation returned %d results for %d groups", kind, len(results), len(groups))
	}
	texts := make(map[string]generatedText, len(chunks))
	missing := make([]rag.Chunk, 0)
	missingDocs := make([]rag.Document, 0)
	for i, group := range groups {
		parsed := parseBatchedResponse(results[i].Text, len(group.chunks))
		for index, chunk := range group.chunks {
			if text, ok := parsed[index+1]; ok {
				texts[chunk.ID] = generatedText{text: text, prompt: spec.Prompt}
			} else {
				missing = append(missing, chunk)
				missingDocs = append(missingDocs, group.document)
			}
		}
	}
	if len(missing) == 0 {
		return texts, nil
	}
	// Repair pass: per-chunk calls with the per-chunk canonical prompt, so a
	// chunk already generated by a per-chunk run is a cache hit.
	repairs := make([]rag.GenerationRequest, len(missing))
	for i, chunk := range missing {
		text := chunk.Text
		if contextual {
			text = contextualInput(missingDocs[i], chunk, 500)
		}
		repairs[i] = rag.GenerationRequest{
			Kind: kind, Model: spec.Model, Prompt: spec.FallbackPrompt, Text: text,
		}
	}
	repaired, err := generate(ctx, repairs)
	if err != nil {
		return nil, errors.Wrapf(err, "repair %d missing batched %s representations", len(missing), kind)
	}
	if len(repaired) != len(missing) {
		return nil, errors.Errorf("batched %s repair returned %d results for %d chunks", kind, len(repaired), len(missing))
	}
	for i, chunk := range missing {
		if text := strings.TrimSpace(repaired[i].Text); text != "" {
			texts[chunk.ID] = generatedText{text: text, prompt: spec.FallbackPrompt}
		}
	}
	return texts, nil
}

// ContextualBatched is the batched counterpart of Contextual.
func ContextualBatched(
	ctx context.Context,
	documents []rag.Document,
	chunks []rag.Chunk,
	generate BatchGenerate,
	spec BatchedSpec,
	includeChunkBody bool,
) ([]rag.Representation, error) {
	texts, err := generateBatched(ctx, documents, chunks, generate, KindContextual, spec, true)
	if err != nil {
		return nil, err
	}
	reps := make([]rag.Representation, 0, len(chunks))
	for _, chunk := range chunks {
		generatedText, ok := texts[chunk.ID]
		if !ok {
			continue
		}
		text := generatedText.text
		if includeChunkBody {
			text = generatedText.text + "\n" + chunk.Text
		}
		rep, err := generated(chunk, KindContextual, text, spec.Model, generatedText.prompt)
		if err != nil {
			return nil, err
		}
		reps = append(reps, rep)
	}
	return reps, nil
}

// GeneratedSummariesBatched is the batched counterpart of GeneratedSummaries.
func GeneratedSummariesBatched(
	ctx context.Context,
	documents []rag.Document,
	chunks []rag.Chunk,
	generate BatchGenerate,
	spec BatchedSpec,
) ([]rag.Representation, error) {
	texts, err := generateBatched(ctx, documents, chunks, generate, KindSummary, spec, false)
	if err != nil {
		return nil, err
	}
	reps := make([]rag.Representation, 0, len(chunks))
	for _, chunk := range chunks {
		generatedText, ok := texts[chunk.ID]
		if !ok {
			continue
		}
		rep, err := generated(chunk, KindSummary, generatedText.text, spec.Model, generatedText.prompt)
		if err != nil {
			return nil, err
		}
		reps = append(reps, rep)
	}
	return reps, nil
}

// GeneratedQuestionsBatched is the batched counterpart of GeneratedQuestions:
// each chunk's text carries up to three questions, one per line.
func GeneratedQuestionsBatched(
	ctx context.Context,
	documents []rag.Document,
	chunks []rag.Chunk,
	generate BatchGenerate,
	spec BatchedSpec,
) ([]rag.Representation, error) {
	texts, err := generateBatched(ctx, documents, chunks, generate, KindQuestion, spec, false)
	if err != nil {
		return nil, err
	}
	reps := make([]rag.Representation, 0, len(chunks)*3)
	for _, chunk := range chunks {
		generatedText := texts[chunk.ID]
		for _, question := range ParseQuestionLines(generatedText.text) {
			rep, err := generated(chunk, KindQuestion, question, spec.Model, generatedText.prompt)
			if err != nil {
				return nil, err
			}
			reps = append(reps, rep)
		}
	}
	return reps, nil
}

// KindStatement is the atomic-statement kind of the decomposition arms (E17):
// each generated factual statement is its own representation hydrating to its
// source chunk — the declarative mirror of KindQuestion.
const KindStatement = "statement"

// GeneratedStatementsBatched builds atomic-statement representations (E17)
// with the same engine as GeneratedQuestionsBatched: one call per ~12-chunk
// group, fence-tolerant JSON parsing, per-chunk repair with the fallback
// prompt. Each chunk's text carries its statements one per line, and every
// line becomes its own representation. ParseQuestionLines is generic over
// lines (bullets, numbering, and fences stripped; everything else verbatim),
// so statements reuse it directly.
func GeneratedStatementsBatched(
	ctx context.Context,
	documents []rag.Document,
	chunks []rag.Chunk,
	generate BatchGenerate,
	spec BatchedSpec,
) ([]rag.Representation, error) {
	texts, err := generateBatched(ctx, documents, chunks, generate, KindStatement, spec, false)
	if err != nil {
		return nil, err
	}
	reps := make([]rag.Representation, 0, len(chunks)*3)
	for _, chunk := range chunks {
		generatedText := texts[chunk.ID]
		for _, statement := range ParseQuestionLines(generatedText.text) {
			rep, err := generated(chunk, KindStatement, statement, spec.Model, generatedText.prompt)
			if err != nil {
				return nil, err
			}
			reps = append(reps, rep)
		}
	}
	return reps, nil
}

// EntityExpansionsBatched is the batched counterpart of EntityExpansions.
func EntityExpansionsBatched(
	ctx context.Context,
	documents []rag.Document,
	chunks []rag.Chunk,
	generate BatchGenerate,
	spec BatchedSpec,
) ([]rag.Representation, error) {
	texts, err := generateBatched(ctx, documents, chunks, generate, KindEntities, spec, false)
	if err != nil {
		return nil, err
	}
	reps := make([]rag.Representation, 0, len(chunks))
	for _, chunk := range chunks {
		generatedText := texts[chunk.ID]
		line := collapseLine(generatedText.text)
		if line == "" {
			continue
		}
		rep, err := generated(chunk, KindEntities, chunk.Text+"\n"+line, spec.Model, generatedText.prompt)
		if err != nil {
			return nil, err
		}
		reps = append(reps, rep)
	}
	return reps, nil
}
