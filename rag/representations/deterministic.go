package representations

import (
	"fmt"

	"github.com/go-go-golems/ragkit/rag"
)

// KindBreadcrumb and KindSmall are the deterministic (free) representation
// kinds of the chunk lab's Track A. A breadcrumb representation carries the
// heading path followed by the chunk body; a small representation carries a
// small chunk's text indexed under its larger parent chunk, so hydration
// performs the small-to-big move with no new code.
const (
	KindBreadcrumb = "breadcrumb"
	KindSmall      = "small"
)

// Breadcrumbs builds one breadcrumb representation per chunk: the markdown
// heading path that governs the chunk's start, prepended as one line to the
// chunk body. This is the free half of contextual retrieval (E3): if it
// captures most of the contextual arm's lift, the generated blurbs are not
// worth their tokens.
func Breadcrumbs(documents []rag.Document, chunks []rag.Chunk) ([]rag.Representation, error) {
	byID := make(map[string]rag.Document, len(documents))
	for _, document := range documents {
		byID[document.ID] = document
	}
	reps := make([]rag.Representation, 0, len(chunks))
	for _, chunk := range chunks {
		document, ok := byID[chunk.DocumentID]
		if !ok {
			return nil, fmt.Errorf("chunk %s references unknown document %q", chunk.ID, chunk.DocumentID)
		}
		rep, err := representation(chunk, KindBreadcrumb, HeadingPath(document, chunk)+"\n"+chunk.Text)
		if err != nil {
			return nil, err
		}
		reps = append(reps, rep)
	}
	return reps, nil
}

// SmallToBig indexes each small chunk's text under its parent chunk: the
// representation's ChunkID names the parent whose byte range overlaps the
// small chunk the most, so a hit over the precise small text hydrates to the
// larger parent (E5). Both chunk sets must come from the same documents.
func SmallToBig(small, parents []rag.Chunk) ([]rag.Representation, error) {
	parentsByDocument := make(map[string][]rag.Chunk)
	for _, parent := range parents {
		parentsByDocument[parent.DocumentID] = append(parentsByDocument[parent.DocumentID], parent)
	}
	reps := make([]rag.Representation, 0, len(small))
	seen := make(map[string]struct{}, len(small))
	for _, chunk := range small {
		parent, ok := bestParent(chunk, parentsByDocument[chunk.DocumentID])
		if !ok {
			return nil, fmt.Errorf("small chunk %s has no parent covering bytes [%d,%d) in document %s",
				chunk.ID, chunk.Range.ByteStart, chunk.Range.ByteEnd, chunk.DocumentID)
		}
		// The identity derives from the parent (whose text the hit will
		// hydrate to) but the searchable text is the small chunk's.
		rep, err := representation(parent, KindSmall, chunk.Text)
		if err != nil {
			return nil, err
		}
		if _, duplicate := seen[rep.ID]; duplicate {
			continue
		}
		seen[rep.ID] = struct{}{}
		reps = append(reps, rep)
	}
	return reps, nil
}

// bestParent picks the parent chunk with the largest byte overlap, breaking
// ties by stable source position and identity rather than caller order.
func bestParent(chunk rag.Chunk, parents []rag.Chunk) (rag.Chunk, bool) {
	best := rag.Chunk{}
	bestOverlap := 0
	for _, parent := range parents {
		start := max(chunk.Range.ByteStart, parent.Range.ByteStart)
		end := min(chunk.Range.ByteEnd, parent.Range.ByteEnd)
		overlap := end - start
		if overlap > bestOverlap || (overlap == bestOverlap && overlap > 0 && parentBefore(parent, best)) {
			best, bestOverlap = parent, overlap
		}
	}
	return best, bestOverlap > 0
}

func parentBefore(left, right rag.Chunk) bool {
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
