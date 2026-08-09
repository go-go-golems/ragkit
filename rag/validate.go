package rag

import (
	"fmt"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/go-go-golems/ragkit/digest"
)

// ValidateHitScores rejects search results that cannot be ranked or serialized
// consistently. It is suitable for generic retrieval transforms that only need
// score/rank closure, even when a synthetic test or local transform does not
// carry every identity field.
func ValidateHitScores(hits []Hit) error {
	for index, hit := range hits {
		if hit.Rank < 1 {
			return fmt.Errorf("hit %d has invalid rank %d", index, hit.Rank)
		}
		if math.IsNaN(hit.Score) || math.IsInf(hit.Score, 0) {
			return fmt.Errorf("hit %d has a non-finite score", index)
		}
	}
	return nil
}

// ValidateHits rejects search results that cannot be serialized or ranked
// consistently. Search backends own identity and ordering details; every
// backend result must nevertheless have finite scores, positive ranks, and
// complete source identities before crossing the answering boundary.
func ValidateHits(hits []Hit) error {
	if err := ValidateHitScores(hits); err != nil {
		return err
	}
	for index, hit := range hits {
		if strings.TrimSpace(hit.RepresentationID) == "" || strings.TrimSpace(hit.ChunkID) == "" || strings.TrimSpace(hit.DocumentID) == "" {
			return fmt.Errorf("hit %d has incomplete identity", index)
		}
	}
	return nil
}

// ValidateDocument checks the invariants common to every corpus loader.
func ValidateDocument(document Document) error {
	if document.ID == "" {
		return fmt.Errorf("document ID is required")
	}
	if !utf8.ValidString(document.Text) {
		return fmt.Errorf("document %q contains invalid UTF-8", document.ID)
	}
	if document.ContentDigest == "" {
		return fmt.Errorf("document %q content digest is required", document.ID)
	}
	if actual := digest.Text(document.Text); actual != document.ContentDigest {
		return fmt.Errorf("document %q content digest mismatch: stored=%s actual=%s", document.ID, document.ContentDigest, actual)
	}
	return nil
}

// ValidateChunk checks identity, source range, and exact source-slice equality.
func ValidateChunk(document Document, chunk Chunk) error {
	if err := ValidateDocument(document); err != nil {
		return err
	}
	if chunk.ID == "" {
		return fmt.Errorf("chunk ID is required")
	}
	if chunk.DocumentID != document.ID {
		return fmt.Errorf("chunk %q belongs to document %q, not %q", chunk.ID, chunk.DocumentID, document.ID)
	}
	if chunk.Range.ByteStart < 0 || chunk.Range.ByteEnd < chunk.Range.ByteStart || chunk.Range.ByteEnd > len(document.Text) {
		return fmt.Errorf("chunk %q has invalid byte range [%d,%d)", chunk.ID, chunk.Range.ByteStart, chunk.Range.ByteEnd)
	}
	if document.Text[chunk.Range.ByteStart:chunk.Range.ByteEnd] != chunk.Text {
		return fmt.Errorf("chunk %q text does not match its source range", chunk.ID)
	}
	if !utf8.ValidString(chunk.Text) {
		return fmt.Errorf("chunk %q contains invalid UTF-8", chunk.ID)
	}
	if chunk.ContentDigest == "" {
		return fmt.Errorf("chunk %q content digest is required", chunk.ID)
	}
	if actual := digest.Text(chunk.Text); actual != chunk.ContentDigest {
		return fmt.Errorf("chunk %q content digest mismatch: stored=%s actual=%s", chunk.ID, chunk.ContentDigest, actual)
	}
	if chunk.Chunker == "" {
		return fmt.Errorf("chunk %q chunker is required", chunk.ID)
	}
	return nil
}

// ValidateCorpus checks source identity, uniqueness, and every chunk's exact
// lineage before a corpus reaches an index backend.
func ValidateCorpus(documents []Document, chunks []Chunk) error {
	documentByID := make(map[string]Document, len(documents))
	for index, document := range documents {
		if err := ValidateDocument(document); err != nil {
			return fmt.Errorf("document %d: %w", index, err)
		}
		if _, exists := documentByID[document.ID]; exists {
			return fmt.Errorf("duplicate document ID %q", document.ID)
		}
		documentByID[document.ID] = document
	}
	chunkIDs := make(map[string]struct{}, len(chunks))
	ordinals := make(map[string]map[int]struct{}, len(documents))
	for index, chunk := range chunks {
		if _, exists := chunkIDs[chunk.ID]; exists {
			return fmt.Errorf("duplicate chunk ID %q", chunk.ID)
		}
		chunkIDs[chunk.ID] = struct{}{}
		document, exists := documentByID[chunk.DocumentID]
		if !exists {
			return fmt.Errorf("chunk %d references unknown document %q", index, chunk.DocumentID)
		}
		if chunk.Ordinal < 0 {
			return fmt.Errorf("chunk %d has negative ordinal %d", index, chunk.Ordinal)
		}
		if ordinals[chunk.DocumentID] == nil {
			ordinals[chunk.DocumentID] = map[int]struct{}{}
		}
		if _, duplicate := ordinals[chunk.DocumentID][chunk.Ordinal]; duplicate {
			return fmt.Errorf("chunk %d duplicates ordinal %d for document %q", index, chunk.Ordinal, chunk.DocumentID)
		}
		ordinals[chunk.DocumentID][chunk.Ordinal] = struct{}{}
		if err := ValidateChunk(document, chunk); err != nil {
			return fmt.Errorf("chunk %d: %w", index, err)
		}
	}
	return nil
}

// ValidateRepresentations checks searchable-text identity and lineage against
// the source chunks. Raw representations must be exact source evidence;
// generated representations carry their own text digest.
func ValidateRepresentations(chunks []Chunk, representations []Representation) error {
	chunkByID := make(map[string]Chunk, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ID] = chunk
	}
	seen := make(map[string]struct{}, len(representations))
	for index, representation := range representations {
		if strings.TrimSpace(representation.ID) == "" {
			return fmt.Errorf("representation %d has no ID", index)
		}
		if _, exists := seen[representation.ID]; exists {
			return fmt.Errorf("duplicate representation ID %q", representation.ID)
		}
		seen[representation.ID] = struct{}{}
		chunk, exists := chunkByID[representation.ChunkID]
		if !exists {
			return fmt.Errorf("representation %q references unknown chunk %q", representation.ID, representation.ChunkID)
		}
		if strings.TrimSpace(representation.Kind) == "" {
			return fmt.Errorf("representation %q has no kind", representation.ID)
		}
		if !utf8.ValidString(representation.Text) {
			return fmt.Errorf("representation %q contains invalid UTF-8", representation.ID)
		}
		if representation.ContentDigest == "" {
			return fmt.Errorf("representation %q content digest is required", representation.ID)
		}
		if actual := digest.Text(representation.Text); actual != representation.ContentDigest {
			return fmt.Errorf("representation %q content digest mismatch: stored=%s actual=%s", representation.ID, representation.ContentDigest, actual)
		}
		if representation.Kind == "raw" && (representation.Text != chunk.Text || representation.ContentDigest != chunk.ContentDigest) {
			return fmt.Errorf("raw representation %q differs from chunk %q", representation.ID, chunk.ID)
		}
	}
	return nil
}

// ValidateQueries requires stable, unique query identities before evaluation
// or aggregation can assign weights to them.
func ValidateQueries(queries []Query) error {
	seen := make(map[string]struct{}, len(queries))
	for index, query := range queries {
		if strings.TrimSpace(query.ID) == "" {
			return fmt.Errorf("query %d has no ID", index)
		}
		if _, exists := seen[query.ID]; exists {
			return fmt.Errorf("duplicate query ID %q", query.ID)
		}
		seen[query.ID] = struct{}{}
	}
	return nil
}
