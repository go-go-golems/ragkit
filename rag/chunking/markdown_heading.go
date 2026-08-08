package chunking

import (
	"context"
	"fmt"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// MarkdownHeading splits at Markdown headings and keeps each section whole
// when it fits the maximum. Sections larger than MaxSectionRunes are split by a
// fixed window with overlap, and sections smaller than MinSectionRunes are
// merged into the following section so a heading never produces a tiny shard.
//
// This is the chunker the assessment (RAG-TTC-ASSESS-001 §4.5) asked for: the
// existing Markdown chunker cuts 90% of chunks at the size limit because it
// re-windows every section. MarkdownHeading lets structure decide the cuts and
// only falls back to a size window for oversized sections.
type MarkdownHeading struct {
	// MaxSectionRunes is the hard ceiling for one chunk. A section larger than
	// this is split by a fixed window.
	MaxSectionRunes int `json:"max_section_runes"`
	// MinSectionRunes is the floor. Sections smaller than this are merged into
	// the next section rather than emitted as a shard.
	MinSectionRunes int `json:"min_section_runes"`
	// OverlapRunes is the overlap applied only to the fixed-window fallback for
	// oversized sections. Whole sections carry no overlap because they end at a
	// structural boundary.
	OverlapRunes int `json:"overlap_runes"`
}

var _ rag.Chunker = (*MarkdownHeading)(nil)

func (chunker *MarkdownHeading) Name() string {
	if chunker == nil {
		return "markdown-heading-invalid"
	}
	return fmt.Sprintf(
		"markdown-heading-%d-%d-%d",
		chunker.MaxSectionRunes, chunker.MinSectionRunes, chunker.OverlapRunes,
	)
}

func (chunker *MarkdownHeading) Chunk(
	ctx context.Context, document rag.Document,
) ([]rag.Chunk, error) {
	if chunker == nil {
		return nil, fmt.Errorf("markdown-heading chunker is nil")
	}
	if err := validateWindow(chunker.MaxSectionRunes, chunker.OverlapRunes); err != nil {
		return nil, err
	}
	if chunker.MinSectionRunes < 0 {
		return nil, fmt.Errorf("minimum section runes must be non-negative")
	}
	if chunker.MinSectionRunes >= chunker.MaxSectionRunes {
		return nil, fmt.Errorf("minimum section runes must be smaller than the maximum")
	}
	if err := rag.ValidateDocument(document); err != nil {
		return nil, err
	}
	sections := markdownSections(document.Text)
	// Merge tiny sections into their successor so a heading over a one-line
	// paragraph does not become its own retrieval shard.
	merged := mergeSmallSections(sections, chunker.MinSectionRunes, document.Text)
	chunks := make([]rag.Chunk, 0, len(merged))
	for _, section := range merged {
		sectionText := document.Text[section.start:section.end]
		var sectionChunks []rag.Chunk
		var err error
		if len([]rune(sectionText)) <= chunker.MaxSectionRunes {
			// The section fits: emit it whole, with no overlap, because it ends
			// at a structural boundary.
			sectionChunks, err = wholeSection(
				ctx, document, section.start, sectionText,
				chunker.Name(), len(chunks),
			)
		} else {
			// The section is oversized: fall back to a fixed window with
			// overlap so a retrieval boundary still lands inside it.
			sectionChunks, err = fixedRange(
				ctx, document, section.start, sectionText,
				chunker.MaxSectionRunes, chunker.OverlapRunes,
				chunker.Name(), len(chunks),
			)
		}
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, sectionChunks...)
	}
	if len(chunks) == 0 && document.Text != "" {
		// A document with no headings and non-empty text still produces one
		// whole-section chunk via the merge path; guard against an empty result
		// from a document that is only whitespace.
		return nil, errors.Errorf("document %q produced no chunks", document.ID)
	}
	return chunks, nil
}

// wholeSection emits one chunk covering an entire section verbatim. It is the
// structural-path counterpart to fixedRange: no windowing, no overlap.
func wholeSection(
	_ context.Context,
	document rag.Document,
	baseByte int,
	text string,
	name string,
	ordinalBase int,
) ([]rag.Chunk, error) {
	if text == "" {
		return nil, nil
	}
	byteStart := baseByte
	byteEnd := baseByte + len(text)
	chunkText := document.Text[byteStart:byteEnd]
	idDigest, err := digest.JSON(struct {
		DocumentID string `json:"document_id"`
		Chunker    string `json:"chunker"`
		ByteStart  int    `json:"byte_start"`
		ByteEnd    int    `json:"byte_end"`
		TextDigest string `json:"text_digest"`
	}{
		DocumentID: document.ID,
		Chunker:    name,
		ByteStart:  byteStart,
		ByteEnd:    byteEnd,
		TextDigest: digest.Text(chunkText),
	})
	if err != nil {
		return nil, err
	}
	chunk := rag.Chunk{
		ID:            "chunk-" + idDigest[:16],
		DocumentID:    document.ID,
		Ordinal:       ordinalBase,
		Range:         rag.Range{ByteStart: byteStart, ByteEnd: byteEnd},
		Text:          chunkText,
		ContentDigest: digest.Text(chunkText),
		Chunker:       name,
	}
	if err := rag.ValidateChunk(document, chunk); err != nil {
		return nil, err
	}
	return []rag.Chunk{chunk}, nil
}

// mergeSmallSections folds each section shorter than minRunes into the next
// section. A trailing small section is merged into its predecessor. The merge
// is structural: it changes byte ranges, not text, so a later fixed-window
// split still produces exact slices of the original document.
func mergeSmallSections(sections []section, minRunes int, text string) []section {
	if minRunes <= 0 || len(sections) <= 1 {
		return sections
	}
	merged := make([]section, 0, len(sections))
	var pending *section
	for index, current := range sections {
		currentRunes := len([]rune(text[current.start:current.end]))
		if currentRunes < minRunes && index < len(sections)-1 {
			if pending == nil {
				copy := current
				pending = &copy
			} else {
				pending.end = current.end
			}
			continue
		}
		if pending != nil {
			current.start = pending.start
			pending = nil
		}
		if currentRunes < minRunes && index == len(sections)-1 && len(merged) > 0 {
			merged[len(merged)-1].end = current.end
			continue
		}
		merged = append(merged, current)
	}
	return merged
}

// sectionDigest, digestText and digestJSON were placeholders during drafting
// and are no longer needed: the whole-section path now uses digest.JSON and
// digest.Text directly, the same helpers fixedRange uses, so chunk ids stay
// comparable across chunkers.

// headingLine reports whether a line is a Markdown heading. Exported as a
// lowercase helper so tests in this package can exercise the boundary.
func headingLine(line string) bool {
	return strings.HasPrefix(strings.TrimLeft(line, " \t"), "#")
}
