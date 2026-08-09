package evaluation

import (
	"fmt"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
)

// Target names the identity level used by relevance judgments. It aliases
// the shared rag.Target vocabulary.
type Target = rag.Target

const (
	TargetRepresentation = rag.TargetRepresentation
	TargetChunk          = rag.TargetChunk
	TargetDocument       = rag.TargetDocument
	TargetUnit           = rag.TargetUnit
)

// TargetResolver resolves RAG hits to evaluation identities.
type TargetResolver struct {
	Documents map[string]rag.Document
}

// HitID returns the hit identity selected by target.
func (resolver TargetResolver) HitID(hit rag.Hit, target Target) (string, error) {
	switch target {
	case TargetRepresentation:
		if strings.TrimSpace(hit.RepresentationID) == "" {
			return "", fmt.Errorf("hit has no representation identity")
		}
		return hit.RepresentationID, nil
	case TargetChunk:
		if strings.TrimSpace(hit.ChunkID) == "" {
			return "", fmt.Errorf("hit has no chunk identity")
		}
		return hit.ChunkID, nil
	case TargetDocument:
		if strings.TrimSpace(hit.DocumentID) == "" {
			return "", fmt.Errorf("hit has no document identity")
		}
		return hit.DocumentID, nil
	case TargetUnit:
		document, ok := resolver.Documents[hit.DocumentID]
		if !ok {
			return "", fmt.Errorf("hit references unknown document %q", hit.DocumentID)
		}
		unit := document.Metadata["evaluation_unit_id"]
		if unit == "" {
			return "", fmt.Errorf("document %q has no evaluation unit identity", hit.DocumentID)
		}
		return unit, nil
	default:
		return "", fmt.Errorf("unsupported relevance target %q", target)
	}
}
