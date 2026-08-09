package representations

import (
	"context"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// KindContextual and KindEntities are the generated representation kinds the
// chunk lab adds beside summary and question. A contextual representation
// carries a situating blurb (optionally followed by the chunk body); an
// entities representation carries the chunk body followed by an expansion line
// of species, common names, and synonyms.
const (
	KindContextual = "contextual"
	KindEntities   = "entities"
)

// BatchGenerate runs a batch of generation requests and returns one result per
// request, aligned by index. The caller owns parallelism, caching, and
// budgets; builders in this package own only prompt assembly and
// representation lineage.
type BatchGenerate func(context.Context, []rag.GenerationRequest) ([]rag.GenerationResult, error)

// GeneratedSpec names one generation configuration: the model identity for the
// cache key and the verbatim instruction prompt. The prompt is part of the
// representation's provenance (PromptDigest), so a changed prompt produces new
// representation ids and new cache entries.
type GeneratedSpec struct {
	Model  string
	Prompt string
}

// ContextualSpec configures the contextual builder. LeadRunes bounds how much
// of the document opening the prompt sees (the "lite" method); zero means the
// default of 500. IncludeChunkBody selects between the contextual arm (blurb +
// chunk text) and the blurb-only arm.
type ContextualSpec struct {
	Model            string
	Prompt           string
	LeadRunes        int
	IncludeChunkBody bool
}

// Contextual builds one contextual representation per chunk: a generated
// situating blurb, prepended to the chunk body when IncludeChunkBody is set.
// The prompt input joins the chunk to its parent document (title, heading
// path, document lead), which is the API gap the intern guide names in
// section 5.3.
func Contextual(
	ctx context.Context,
	documents []rag.Document,
	chunks []rag.Chunk,
	generate BatchGenerate,
	spec ContextualSpec,
) ([]rag.Representation, error) {
	if generate == nil {
		return nil, errors.New("the contextual builder needs a generation function")
	}
	byID := make(map[string]rag.Document, len(documents))
	for _, document := range documents {
		byID[document.ID] = document
	}
	lead := spec.LeadRunes
	if lead <= 0 {
		lead = 500
	}
	requests := make([]rag.GenerationRequest, len(chunks))
	for i, chunk := range chunks {
		document, ok := byID[chunk.DocumentID]
		if !ok {
			return nil, errors.Errorf("chunk %s references unknown document %q", chunk.ID, chunk.DocumentID)
		}
		requests[i] = rag.GenerationRequest{
			Kind: KindContextual, Model: spec.Model,
			Prompt: spec.Prompt, Text: contextualInput(document, chunk, lead),
		}
	}
	results, err := generate(ctx, requests)
	if err != nil {
		return nil, errors.Wrap(err, "generate contextual blurbs")
	}
	if len(results) != len(chunks) {
		return nil, errors.Errorf("contextual generation returned %d results for %d chunks", len(results), len(chunks))
	}
	reps := make([]rag.Representation, 0, len(chunks))
	for i, chunk := range chunks {
		blurb := strings.TrimSpace(results[i].Text)
		if blurb == "" {
			continue
		}
		text := blurb
		if spec.IncludeChunkBody {
			text = blurb + "\n" + chunk.Text
		}
		rep, err := generated(chunk, KindContextual, text, spec.Model, spec.Prompt)
		if err != nil {
			return nil, err
		}
		reps = append(reps, rep)
	}
	return reps, nil
}

// GeneratedSummaries builds one summary representation per chunk from a
// provider generation. It is the batch analogue of Summaries with a
// GeneratorSummarizer: same kind, same lineage, but the caller's BatchGenerate
// supplies parallelism and caching.
func GeneratedSummaries(
	ctx context.Context,
	chunks []rag.Chunk,
	generate BatchGenerate,
	spec GeneratedSpec,
) ([]rag.Representation, error) {
	texts, err := generatePerChunk(ctx, chunks, generate, KindSummary, spec)
	if err != nil {
		return nil, err
	}
	reps := make([]rag.Representation, 0, len(chunks))
	for i, chunk := range chunks {
		text := strings.TrimSpace(texts[i])
		if text == "" {
			continue
		}
		rep, err := generated(chunk, KindSummary, text, spec.Model, spec.Prompt)
		if err != nil {
			return nil, err
		}
		reps = append(reps, rep)
	}
	return reps, nil
}

// GeneratedQuestions builds question representations from one generation per
// chunk. The response is parsed one question per line (bullets and numbering
// stripped), and each question becomes its own representation pointing at the
// source chunk — the same shape Questions produces from a Questioner.
func GeneratedQuestions(
	ctx context.Context,
	chunks []rag.Chunk,
	generate BatchGenerate,
	spec GeneratedSpec,
) ([]rag.Representation, error) {
	texts, err := generatePerChunk(ctx, chunks, generate, KindQuestion, spec)
	if err != nil {
		return nil, err
	}
	reps := make([]rag.Representation, 0, len(chunks)*3)
	for i, chunk := range chunks {
		for _, question := range ParseQuestionLines(texts[i]) {
			rep, err := generated(chunk, KindQuestion, question, spec.Model, spec.Prompt)
			if err != nil {
				return nil, err
			}
			reps = append(reps, rep)
		}
	}
	return reps, nil
}

// EntityExpansions builds one entities representation per chunk: the chunk
// body followed by a generated expansion line of species, common names, and
// synonyms. The chunk body stays in the text so the arm isolates the effect of
// adding the line, not of replacing the chunk.
func EntityExpansions(
	ctx context.Context,
	chunks []rag.Chunk,
	generate BatchGenerate,
	spec GeneratedSpec,
) ([]rag.Representation, error) {
	texts, err := generatePerChunk(ctx, chunks, generate, KindEntities, spec)
	if err != nil {
		return nil, err
	}
	reps := make([]rag.Representation, 0, len(chunks))
	for i, chunk := range chunks {
		line := collapseLine(texts[i])
		if line == "" {
			continue
		}
		rep, err := generated(chunk, KindEntities, chunk.Text+"\n"+line, spec.Model, spec.Prompt)
		if err != nil {
			return nil, err
		}
		reps = append(reps, rep)
	}
	return reps, nil
}

// HeadingPath computes the markdown heading path that governs the chunk's
// start position: the document title followed by the most recent heading at
// each level above the chunk. It is deterministic and free — the same
// breadcrumb the E3 experiment indexes on its own.
func HeadingPath(document rag.Document, chunk rag.Chunk) string {
	start := chunk.Range.ByteStart
	if start > len(document.Text) {
		start = len(document.Text)
	}
	levels := make([]string, 7)
	for _, line := range strings.Split(document.Text[:start], "\n") {
		trimmed := strings.TrimSpace(line)
		level := 0
		for level < len(trimmed) && trimmed[level] == '#' {
			level++
		}
		if level == 0 || level > 6 || level >= len(trimmed) || trimmed[level] != ' ' {
			continue
		}
		levels[level] = strings.TrimSpace(trimmed[level+1:])
		for deeper := level + 1; deeper <= 6; deeper++ {
			levels[deeper] = ""
		}
	}
	path := make([]string, 0, 4)
	if title := strings.TrimSpace(document.Title); title != "" {
		path = append(path, title)
	}
	for level := 1; level <= 6; level++ {
		if levels[level] != "" {
			if level == 1 && len(path) > 0 && path[len(path)-1] == levels[level] {
				continue
			}
			path = append(path, levels[level])
		}
	}
	if len(path) == 0 {
		return "(document root)"
	}
	return strings.Join(path, " > ")
}

// ParseQuestionLines extracts questions from a generated response: one per
// line, with list bullets, numbering, and code fences stripped. Blank lines
// and fence markers vanish; everything else survives verbatim.
func ParseQuestionLines(text string) []string {
	questions := make([]string, 0, 4)
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		line = strings.TrimLeft(line, "-*• \t")
		line = stripOrderedListPrefix(line)
		if line == "" || strings.HasPrefix(line, "```") {
			continue
		}
		questions = append(questions, line)
	}
	return questions
}

func stripOrderedListPrefix(line string) string {
	index := 0
	for index < len(line) && line[index] >= '0' && line[index] <= '9' {
		index++
	}
	if index == 0 || index == len(line) || (line[index] != '.' && line[index] != ')') {
		return line
	}
	return strings.TrimLeft(line[index+1:], " \t")
}

// generatePerChunk sends one request per chunk and returns the raw response
// texts aligned to the chunk slice.
func generatePerChunk(
	ctx context.Context,
	chunks []rag.Chunk,
	generate BatchGenerate,
	kind string,
	spec GeneratedSpec,
) ([]string, error) {
	if generate == nil {
		return nil, errors.Errorf("the %s builder needs a generation function", kind)
	}
	requests := make([]rag.GenerationRequest, len(chunks))
	for i, chunk := range chunks {
		requests[i] = rag.GenerationRequest{
			Kind: kind, Model: spec.Model, Prompt: spec.Prompt, Text: chunk.Text,
		}
	}
	results, err := generate(ctx, requests)
	if err != nil {
		return nil, errors.Wrapf(err, "generate %s representations", kind)
	}
	if len(results) != len(chunks) {
		return nil, errors.Errorf("%s generation returned %d results for %d chunks", kind, len(results), len(chunks))
	}
	texts := make([]string, len(results))
	for i, result := range results {
		texts[i] = result.Text
	}
	return texts, nil
}

// generated builds one representation with provenance: the generating model
// and the prompt digest travel with the representation, so a bundle built from
// lab output records how its retrieval material came to be.
func generated(chunk rag.Chunk, kind, text, model, prompt string) (rag.Representation, error) {
	return representationWithProvenance(chunk, kind, text, model, digest.Text(prompt))
}

// contextualInput renders the per-chunk contextual prompt input. The batched
// engine's repair fallback renders the identical string, so a repair call is a
// cache hit for anything a per-chunk run already generated.
func contextualInput(document rag.Document, chunk rag.Chunk, lead int) string {
	var input strings.Builder
	input.WriteString("Document title: ")
	input.WriteString(document.Title)
	input.WriteString("\nSection path: ")
	input.WriteString(HeadingPath(document, chunk))
	input.WriteString("\nDocument lead:\n")
	input.WriteString(leadRunes(document.Text, lead))
	input.WriteString("\n\nChunk:\n")
	input.WriteString(chunk.Text)
	return input.String()
}

// leadRunes returns the first max runes of text, cut on a rune boundary.
func leadRunes(text string, max int) string {
	runes := []rune(text)
	if len(runes) <= max {
		return text
	}
	return string(runes[:max])
}

// collapseLine flattens a generated response to one line: newlines become
// separators, surrounding whitespace goes away.
func collapseLine(text string) string {
	fields := strings.FieldsFunc(text, func(r rune) bool { return r == '\n' || r == '\r' })
	parts := make([]string, 0, len(fields))
	for _, field := range fields {
		field = strings.TrimSpace(field)
		if field == "" || strings.HasPrefix(field, "```") {
			continue
		}
		parts = append(parts, field)
	}
	return strings.Join(parts, "; ")
}
