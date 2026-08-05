package chunking

import (
	"context"
	"fmt"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
)

// Fixed splits documents by rune count with rune overlap while recording exact
// byte ranges.
type Fixed struct {
	SizeRunes    int `json:"size_runes"`
	OverlapRunes int `json:"overlap_runes"`
}

var _ rag.Chunker = (*Fixed)(nil)

// Name returns the stable chunker identity.
func (chunker *Fixed) Name() string {
	if chunker == nil {
		return "fixed-invalid"
	}
	return fmt.Sprintf("fixed-%d-%d", chunker.SizeRunes, chunker.OverlapRunes)
}

// Chunk splits one document.
func (chunker *Fixed) Chunk(ctx context.Context, document rag.Document) ([]rag.Chunk, error) {
	if chunker == nil {
		return nil, fmt.Errorf("fixed chunker is nil")
	}
	if err := validateWindow(chunker.SizeRunes, chunker.OverlapRunes); err != nil {
		return nil, err
	}
	if err := rag.ValidateDocument(document); err != nil {
		return nil, err
	}
	return fixedRange(ctx, document, 0, document.Text, chunker.SizeRunes, chunker.OverlapRunes, chunker.Name(), 0)
}

func fixedRange(
	ctx context.Context,
	document rag.Document,
	baseByte int,
	text string,
	sizeRunes int,
	overlapRunes int,
	name string,
	ordinalBase int,
) ([]rag.Chunk, error) {
	return fixedRangeSnap(ctx, document, baseByte, text, sizeRunes, overlapRunes, 0, name, ordinalBase)
}

// fixedRangeSnap is fixedRange with optional sentence snapping: when snapRunes
// is positive, a non-final window boundary retreats to the last sentence end
// (., !, ?, or newline) within snapRunes runes, so chunks stop cutting
// mid-sentence. A snap that would stall the window (end at or before start +
// overlap) is reverted, which guarantees progress.
func fixedRangeSnap(
	ctx context.Context,
	document rag.Document,
	baseByte int,
	text string,
	sizeRunes int,
	overlapRunes int,
	snapRunes int,
	name string,
	ordinalBase int,
) ([]rag.Chunk, error) {
	byteOffsets := make([]int, 0, len(text)+1)
	for byteOffset := range text {
		byteOffsets = append(byteOffsets, byteOffset)
	}
	byteOffsets = append(byteOffsets, len(text))
	if len(byteOffsets) == 1 {
		return []rag.Chunk{}, nil
	}

	chunks := make([]rag.Chunk, 0)
	for runeStart := 0; runeStart < len(byteOffsets)-1; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		runeEnd := min(runeStart+sizeRunes, len(byteOffsets)-1)
		if snapRunes > 0 && runeEnd < len(byteOffsets)-1 {
			runeEnd = snapToSentence(text, byteOffsets, runeStart, runeEnd, overlapRunes, snapRunes)
		}
		byteStart := baseByte + byteOffsets[runeStart]
		byteEnd := baseByte + byteOffsets[runeEnd]
		chunkText := document.Text[byteStart:byteEnd]
		ordinal := ordinalBase + len(chunks)
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
			Ordinal:       ordinal,
			Range:         rag.Range{ByteStart: byteStart, ByteEnd: byteEnd},
			Text:          chunkText,
			ContentDigest: digest.Text(chunkText),
			Chunker:       name,
		}
		if err := rag.ValidateChunk(document, chunk); err != nil {
			return nil, err
		}
		chunks = append(chunks, chunk)
		if runeEnd == len(byteOffsets)-1 {
			break
		}
		runeStart = runeEnd - overlapRunes
	}
	return chunks, nil
}

// snapToSentence retreats a window end to just after the last sentence-final
// rune (., !, ?, newline) within snapRunes of the proposed end. It returns the
// proposed end unchanged when no boundary exists in the window or when the
// snap would stall the sliding window.
func snapToSentence(text string, byteOffsets []int, runeStart, runeEnd, overlapRunes, snapRunes int) int {
	lowest := runeEnd - snapRunes
	if lowest < runeStart+1 {
		lowest = runeStart + 1
	}
	for candidate := runeEnd; candidate > lowest; candidate-- {
		switch text[byteOffsets[candidate-1]:byteOffsets[candidate]] {
		case ".", "!", "?", "\n":
			if candidate <= runeStart+overlapRunes {
				return runeEnd
			}
			return candidate
		}
	}
	return runeEnd
}

func validateWindow(sizeRunes, overlapRunes int) error {
	if sizeRunes < 1 {
		return fmt.Errorf("chunk size must be positive")
	}
	if overlapRunes < 0 || overlapRunes >= sizeRunes {
		return fmt.Errorf("chunk overlap must be non-negative and smaller than size")
	}
	return nil
}

// Apply chunks documents in input order and returns one flat ordered slice.
func Apply(ctx context.Context, chunker rag.Chunker, documents []rag.Document) ([]rag.Chunk, error) {
	if chunker == nil {
		return nil, fmt.Errorf("chunker is required")
	}
	var chunks []rag.Chunk
	for _, document := range documents {
		documentChunks, err := chunker.Chunk(ctx, document)
		if err != nil {
			return nil, fmt.Errorf("chunk document %q: %w", document.ID, err)
		}
		chunks = append(chunks, documentChunks...)
	}
	return chunks, nil
}
