package rag

import "fmt"

// Target names an identity level of retrieval material. Retrieval collapse
// and evaluation judgment resolution both select by target; they share this
// definition so the vocabulary cannot drift (retrieval accepts chunk and
// document; evaluation additionally accepts representation and unit).
type Target string

const (
	TargetRepresentation Target = "representation"
	TargetChunk          Target = "chunk"
	TargetDocument       Target = "document"
	TargetUnit           Target = "unit"
)

// Validate rejects missing and unsupported retrieval/evaluation identity
// levels against the one shared target vocabulary.
func (target Target) Validate() error {
	if target == "" {
		return fmt.Errorf("relevance target is required")
	}
	switch target {
	case TargetRepresentation, TargetChunk, TargetDocument, TargetUnit:
		return nil
	default:
		return fmt.Errorf("unsupported relevance target %q", target)
	}
}
