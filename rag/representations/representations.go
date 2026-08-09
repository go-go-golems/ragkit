// Package representations generates non-raw searchable representations of
// chunks: summaries and synthetic questions. These are retrieval material, not
// final evidence: a hit over a summary must hydrate back to its source chunk
// before it is shown to a generator or evaluated (see rag.Representation).
//
// This is the enrichment axis RAG-TTC-ASSESS-001 §4.6 asked for. The bundle
// today carries only raw representations; this package lets an experiment
// generate summary and question representations, embed them, and retrieve over
// them while keeping the source-chunk lineage intact.
package representations

import (
	"context"
	"fmt"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/go-go-golems/ragkit/rag/generation"
	"github.com/pkg/errors"
)

// KindSummary and KindQuestion are the non-raw representation kinds. They match
// the Kind strings the retrieval and index layers already store, so an index
// built over mixed kinds records them without a schema change.
const (
	KindSummary  = "summary"
	KindQuestion = "question"
)

// Summarizer turns one chunk into a summary representation. A real summarizer
// uses a generation provider; a deterministic one is provided for zero-cost
// experiments.
type Summarizer interface {
	Summarize(ctx context.Context, chunk rag.Chunk) (string, error)
}

// Questioner turns one chunk into one or more synthetic retrieval questions.
type Questioner interface {
	Questions(ctx context.Context, chunk rag.Chunk) ([]string, error)
}

// FromChunks builds one representation per chunk for the given kind, using the
// provided text function. The representation's ChunkID always points back to
// the source chunk, which is what makes hydration possible.
func FromChunks(
	ctx context.Context,
	chunks []rag.Chunk,
	kind string,
	text func(context.Context, rag.Chunk) (string, error),
) ([]rag.Representation, error) {
	if kind == "" || kind == "raw" {
		return nil, fmt.Errorf("FromChunks is for non-raw kinds, got %q", kind)
	}
	result := make([]rag.Representation, 0, len(chunks))
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		body, err := text(ctx, chunk)
		if err != nil {
			return nil, errors.Wrapf(err, "generate %s for chunk %s", kind, chunk.ID)
		}
		body = strings.TrimSpace(body)
		if body == "" {
			continue
		}
		rep, err := representation(chunk, kind, body)
		if err != nil {
			return nil, err
		}
		result = append(result, rep)
	}
	return result, nil
}

// representation builds one representation with a stable id derived from the
// chunk, kind, and generated text. The id includes the text digest so a changed
// summary produces a different id (and therefore a different cache key).
func representation(chunk rag.Chunk, kind, text string) (rag.Representation, error) {
	return representationWithProvenance(chunk, kind, text, "", "")
}

// representationWithProvenance is the single identity constructor for
// derived retrieval material. Every field that distinguishes an experiment
// arm participates in the ID before the representation is returned.
func representationWithProvenance(chunk rag.Chunk, kind, text, model, promptDigest string) (rag.Representation, error) {
	identity, err := digest.JSON(struct {
		ChunkID       string `json:"chunk_id"`
		ContentDigest string `json:"content_digest"`
		Kind          string `json:"kind"`
		TextDigest    string `json:"text_digest"`
		Model         string `json:"model,omitempty"`
		PromptDigest  string `json:"prompt_digest,omitempty"`
	}{
		ChunkID: chunk.ID, ContentDigest: chunk.ContentDigest,
		Kind: kind, TextDigest: digest.Text(text), Model: model, PromptDigest: promptDigest,
	})
	if err != nil {
		return rag.Representation{}, fmt.Errorf("build representation identity: %w", err)
	}
	return rag.Representation{
		ID: "rep-" + identity[:16], ChunkID: chunk.ID, Kind: kind,
		Text: text, ContentDigest: digest.Text(text), Model: model, PromptDigest: promptDigest,
	}, nil
}

// GeneratorSummarizer adapts a rag.Generator to the Summarizer interface. It
// sends the chunk text as the generation input and returns the generated text.
// The prompt is the caller's responsibility: a summary prompt for summaries, a
// question-generation prompt for questions.
type GeneratorSummarizer struct {
	Generator rag.Generator
	Model     string
	Prompt    string
}

var _ Summarizer = (*GeneratorSummarizer)(nil)

func (s GeneratorSummarizer) Summarize(ctx context.Context, chunk rag.Chunk) (string, error) {
	if s.Generator == nil {
		return "", fmt.Errorf("GeneratorSummarizer requires a generator")
	}
	result, err := s.Generator.Generate(ctx, rag.GenerationRequest{
		Kind: KindSummary, Model: s.Model, Prompt: s.Prompt, Text: chunk.Text,
	})
	if err != nil {
		return "", err
	}
	return result.Text, nil
}

// ExtractiveSummarizer is a deterministic summarizer for zero-cost experiments.
// It produces a lead-sentence summary: the first sentence (or the first
// maxRunes runes if no sentence boundary is found) of the chunk. It is the
// representation analogue of generation.Extractive.
type ExtractiveSummarizer struct {
	MaxRunes int
}

var _ Summarizer = (*ExtractiveSummarizer)(nil)

func (s ExtractiveSummarizer) Summarize(_ context.Context, chunk rag.Chunk) (string, error) {
	text := strings.TrimSpace(chunk.Text)
	if text == "" {
		return "", nil
	}
	max := s.MaxRunes
	if max <= 0 {
		max = 300
	}
	runes := []rune(text)
	// Lead sentence: up to the first period, question mark, or newline, capped
	// at maxRunes. Both the boundary and the cap are rune offsets.
	for index, current := range runes {
		if index >= max {
			break
		}
		if index > 0 && strings.ContainsRune(".?!\n", current) {
			return string(runes[:index+1]), nil
		}
	}
	if len(runes) <= max {
		return text, nil
	}
	if max == 1 {
		return "…", nil
	}
	return string(runes[:max-1]) + "…", nil
}

// Summaries is a convenience that builds summary representations for every
// chunk using a summarizer.
func Summaries(ctx context.Context, chunks []rag.Chunk, summarizer Summarizer) ([]rag.Representation, error) {
	if summarizer == nil {
		return nil, errors.New("summarizer is required")
	}
	return FromChunks(ctx, chunks, KindSummary, summarizer.Summarize)
}

// Questions builds one question representation per generated question. A chunk
// that yields several questions produces several representations, all pointing
// at the same source chunk.
func Questions(ctx context.Context, chunks []rag.Chunk, questioner Questioner) ([]rag.Representation, error) {
	if questioner == nil {
		return nil, errors.New("questioner is required")
	}
	result := make([]rag.Representation, 0, len(chunks))
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		questions, err := questioner.Questions(ctx, chunk)
		if err != nil {
			return nil, errors.Wrapf(err, "generate questions for chunk %s", chunk.ID)
		}
		for _, question := range uniqueNonemptyLines(questions) {
			rep, err := representation(chunk, KindQuestion, question)
			if err != nil {
				return nil, err
			}
			result = append(result, rep)
		}
	}
	return result, nil
}

func uniqueNonemptyLines(lines []string) []string {
	seen := make(map[string]struct{}, len(lines))
	result := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if _, exists := seen[line]; exists {
			continue
		}
		seen[line] = struct{}{}
		result = append(result, line)
	}
	return result
}

// Compose concatenates representation slices, deduplicating by id so a chunk
// that appears in both a raw and a summary set contributes one of each kind
// rather than colliding.
func Compose(sets ...[]rag.Representation) []rag.Representation {
	seen := make(map[string]struct{})
	combined := make([]rag.Representation, 0)
	for _, set := range sets {
		for _, rep := range set {
			if _, ok := seen[rep.ID]; ok {
				continue
			}
			seen[rep.ID] = struct{}{}
			combined = append(combined, rep)
		}
	}
	return combined
}

// Hydrate maps non-raw hits back to their source chunks. It is the enforcement
// of the rule that a summary or question representation is retrieval material,
// never final evidence: a hit over a summary becomes the source chunk before it
// reaches a generator or an evaluator.
func Hydrate(hits []rag.Hit, chunks []rag.Chunk) ([]rag.Chunk, error) {
	byID := make(map[string]rag.Chunk, len(chunks))
	for _, chunk := range chunks {
		byID[chunk.ID] = chunk
	}
	result := make([]rag.Chunk, 0, len(hits))
	for _, hit := range hits {
		chunk, ok := byID[hit.ChunkID]
		if !ok {
			return nil, fmt.Errorf("hit references unknown chunk %q", hit.ChunkID)
		}
		result = append(result, chunk)
	}
	return result, nil
}

// EnsureRaw ensures a representation set contains a raw representation for
// every chunk. Enrichment adds to raw; it does not replace it, because raw is
// the only kind that is its own evidence.
func EnsureRaw(chunks []rag.Chunk, reps []rag.Representation) []rag.Representation {
	haveRaw := make(map[string]struct{}, len(reps))
	for _, rep := range reps {
		if rep.Kind == "raw" {
			haveRaw[rep.ChunkID] = struct{}{}
		}
	}
	missing := make([]rag.Representation, 0)
	for _, chunk := range chunks {
		if _, ok := haveRaw[chunk.ID]; ok {
			continue
		}
		missing = append(missing, generation.Raw([]rag.Chunk{chunk})...)
	}
	if len(missing) == 0 {
		return reps
	}
	return Compose(reps, missing)
}
