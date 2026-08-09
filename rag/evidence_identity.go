package rag

import (
	"strings"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/pkg/errors"
)

// EvidenceIdentity is the score-independent semantic identity of one ordered
// evidence item. Slice order carries evidence order; backend and reranker
// scores are observations and do not affect the source material presented to
// a provider or reviewer.
type EvidenceIdentity struct {
	ChunkID       string `json:"chunk_id"`
	ContentDigest string `json:"content_digest"`
}

// EvidenceIdentities returns the ordered semantic identities for evidence.
// ContentDigest is derived from chunk text when older or synthetic records do
// not provide it explicitly.
func EvidenceIdentities(evidence []Evidence) ([]EvidenceIdentity, error) {
	ret := make([]EvidenceIdentity, len(evidence))
	for index, item := range evidence {
		if strings.TrimSpace(item.Chunk.ID) == "" {
			return nil, errors.Errorf("evidence %d has no chunk ID", index)
		}
		contentDigest := item.Chunk.ContentDigest
		actualDigest := digest.Text(item.Chunk.Text)
		if contentDigest == "" {
			contentDigest = actualDigest
		} else if contentDigest != actualDigest {
			return nil, errors.Errorf("evidence %d chunk %q content digest mismatch: stored=%s actual=%s", index, item.Chunk.ID, contentDigest, actualDigest)
		}
		ret[index] = EvidenceIdentity{
			ChunkID:       item.Chunk.ID,
			ContentDigest: contentDigest,
		}
	}
	return ret, nil
}
