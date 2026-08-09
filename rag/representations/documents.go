package representations

import (
	"context"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// KindDocumentSummary is the document-level summary kind of the raptor-lite
// arm (E16): one generated summary per whole document, indexed beside the
// chunk-level summaries so queries whose answer spans chunks can hit either
// level.
const KindDocumentSummary = "document-summary"

// DocumentSummaries builds one summary representation per document from one
// generation call per document. The prompt input is the full document text —
// the corpus documents are small, so no truncation is needed or wanted: a
// truncated input would summarize a different text than the one indexed.
//
// Decision to record (E16, intern guide): each document-level representation
// hydrates to the document's FIRST chunk. Evidence must remain an exact
// chunk — a whole document is not admissible evidence — so a hit over a
// document summary becomes the document's opening chunk, the closest exact
// slice to "the document as a whole".
func DocumentSummaries(
	ctx context.Context,
	documents []rag.Document,
	chunks []rag.Chunk,
	generate BatchGenerate,
	spec GeneratedSpec,
) ([]rag.Representation, error) {
	if generate == nil {
		return nil, errors.New("the document summary builder needs a generation function")
	}
	// First chunk per document by source order. Callers may combine or shuffle
	// chunk slices, so input order is not a reliable hydration boundary.
	firstChunk := make(map[string]rag.Chunk, len(documents))
	for _, chunk := range chunks {
		current, ok := firstChunk[chunk.DocumentID]
		if !ok || chunkBefore(chunk, current) {
			firstChunk[chunk.DocumentID] = chunk
		}
	}
	for _, document := range documents {
		if _, ok := firstChunk[document.ID]; !ok {
			return nil, errors.Errorf("document %q has no chunks to hydrate its summary to", document.ID)
		}
	}
	requests := make([]rag.GenerationRequest, len(documents))
	for i, document := range documents {
		requests[i] = rag.GenerationRequest{
			Kind: KindDocumentSummary, Model: spec.Model, Prompt: spec.Prompt,
			Text: document.Text,
		}
	}
	results, err := generate(ctx, requests)
	if err != nil {
		return nil, errors.Wrap(err, "generate document summaries")
	}
	if len(results) != len(documents) {
		return nil, errors.Errorf("document summary generation returned %d results for %d documents", len(results), len(documents))
	}
	reps := make([]rag.Representation, 0, len(documents))
	for i, document := range documents {
		text := strings.TrimSpace(results[i].Text)
		if text == "" {
			continue
		}
		chunk := firstChunk[document.ID]
		rep, err := generated(chunk, KindDocumentSummary, text, spec.Model, spec.Prompt)
		if err != nil {
			return nil, err
		}
		reps = append(reps, rep)
	}
	return reps, nil
}

func chunkBefore(left, right rag.Chunk) bool {
	if left.Ordinal != right.Ordinal {
		return left.Ordinal < right.Ordinal
	}
	if left.Range.ByteStart != right.Range.ByteStart {
		return left.Range.ByteStart < right.Range.ByteStart
	}
	if left.Range.ByteEnd != right.Range.ByteEnd {
		return left.Range.ByteEnd < right.Range.ByteEnd
	}
	return left.ID < right.ID
}
