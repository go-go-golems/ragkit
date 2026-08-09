package chunking

import (
	"context"
	"fmt"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
)

// Markdown splits at Markdown headings and applies a fixed window to oversized
// sections. When SentenceSnapRunes is positive, non-final window boundaries
// retreat to the last sentence end within that many runes (the E4
// sentence-snap experiment); the name records the parameter because it changes
// the chunks and therefore every digest built on them.
type Markdown struct {
	MaxSectionRunes   int `json:"max_section_runes"`
	OverlapRunes      int `json:"overlap_runes"`
	SentenceSnapRunes int `json:"sentence_snap_runes,omitempty"`
}

var _ rag.Chunker = (*Markdown)(nil)

func (chunker *Markdown) Name() string {
	if chunker == nil {
		return "markdown-invalid"
	}
	if chunker.SentenceSnapRunes > 0 {
		return fmt.Sprintf("markdown-%d-%d-snap%d",
			chunker.MaxSectionRunes, chunker.OverlapRunes, chunker.SentenceSnapRunes)
	}
	return fmt.Sprintf("markdown-%d-%d", chunker.MaxSectionRunes, chunker.OverlapRunes)
}

func (chunker *Markdown) Chunk(ctx context.Context, document rag.Document) ([]rag.Chunk, error) {
	if chunker == nil {
		return nil, fmt.Errorf("markdown chunker is nil")
	}
	if err := validateWindow(chunker.MaxSectionRunes, chunker.OverlapRunes); err != nil {
		return nil, err
	}
	if err := rag.ValidateDocument(document); err != nil {
		return nil, err
	}
	sections := markdownSections(document.Text)
	chunks := make([]rag.Chunk, 0, len(sections))
	for _, section := range sections {
		sectionChunks, err := fixedRangeSnap(
			ctx,
			document,
			section.start,
			document.Text[section.start:section.end],
			chunker.MaxSectionRunes,
			chunker.OverlapRunes,
			chunker.SentenceSnapRunes,
			chunker.Name(),
			len(chunks),
		)
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, sectionChunks...)
	}
	return chunks, nil
}

type section struct {
	start int
	end   int
}

func markdownSections(text string) []section {
	if text == "" {
		return nil
	}
	starts := []int{0}
	offset := 0
	var fence byte
	var fenceWidth int
	for _, line := range strings.SplitAfter(text, "\n") {
		marker, width, closing := markdownFence(line, fence, fenceWidth)
		if fence != 0 {
			if closing {
				fence, fenceWidth = 0, 0
			}
			offset += len(line)
			continue
		}
		if marker != 0 {
			fence, fenceWidth = marker, width
			offset += len(line)
			continue
		}
		if offset > 0 && isATXHeading(line) {
			starts = append(starts, offset)
		}
		offset += len(line)
	}
	sections := make([]section, 0, len(starts))
	for index, start := range starts {
		end := len(text)
		if index+1 < len(starts) {
			end = starts[index+1]
		}
		sections = append(sections, section{start: start, end: end})
	}
	return sections
}

func isATXHeading(line string) bool {
	line = strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
	indent := 0
	for indent < len(line) && line[indent] == ' ' {
		indent++
	}
	if indent > 3 || indent == len(line) || line[indent] != '#' {
		return false
	}
	end := indent
	for end < len(line) && line[end] == '#' {
		end++
	}
	level := end - indent
	return level >= 1 && level <= 6 && (end == len(line) || line[end] == ' ' || line[end] == '\t')
}

// markdownFence recognizes CommonMark-style backtick and tilde fences with
// up to three leading spaces. While a fence is open, heading-looking code is
// never structural input.
func markdownFence(line string, open byte, openWidth int) (marker byte, width int, closing bool) {
	line = strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
	indent := 0
	for indent < len(line) && line[indent] == ' ' {
		indent++
	}
	if indent > 3 || indent == len(line) {
		return 0, 0, false
	}
	marker = line[indent]
	if marker != '`' && marker != '~' {
		return 0, 0, false
	}
	end := indent
	for end < len(line) && line[end] == marker {
		end++
	}
	width = end - indent
	if width < 3 {
		return 0, 0, false
	}
	if open == 0 {
		return marker, width, false
	}
	if marker == open && width >= openWidth && strings.TrimSpace(line[end:]) == "" {
		return marker, width, true
	}
	return 0, 0, false
}
