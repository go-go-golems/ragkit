package rag

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
