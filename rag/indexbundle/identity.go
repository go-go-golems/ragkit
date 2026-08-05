package indexbundle

import (
	"sort"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

type identity struct {
	SchemaVersion        int             `json:"schema_version"`
	CorpusDigest         string          `json:"corpus_digest"`
	Chunker              ChunkerIdentity `json:"chunker"`
	RepresentationKinds  []string        `json:"representation_kinds"`
	RepresentationDigest string          `json:"representation_digest"`
	Lexical              BackendIdentity `json:"lexical"`
	Vector               *VectorIdentity `json:"vector,omitempty"`
}

func CalculateID(
	documents []rag.Document,
	representations []rag.Representation,
	chunker ChunkerIdentity,
	lexical BackendIdentity,
	vector *VectorIdentity,
) (string, string, string, []string, error) {
	corpusDigest, err := digest.JSON(documents)
	if err != nil {
		return "", "", "", nil, errors.Wrap(err, "digest bundle corpus")
	}
	representationDigest, err := digest.JSON(representations)
	if err != nil {
		return "", "", "", nil, errors.Wrap(err, "digest bundle representations")
	}
	kinds := representationKinds(representations)
	vectorIdentity := cloneVectorIdentity(vector)
	if vectorIdentity != nil {
		vectorIdentity.RepresentationDigest = representationDigest
	}
	id, err := calculateIDFromDigests(
		corpusDigest, representationDigest, kinds, chunker, lexical, vectorIdentity,
	)
	if err != nil {
		return "", "", "", nil, err
	}
	return id, corpusDigest, representationDigest, kinds, nil
}

func calculateIDFromDigests(
	corpusDigest string,
	representationDigest string,
	kinds []string,
	chunker ChunkerIdentity,
	lexical BackendIdentity,
	vector *VectorIdentity,
) (string, error) {
	id, err := digest.TruncatedJSON("rk-", 16, identity{
		SchemaVersion:        SchemaVersion,
		CorpusDigest:         corpusDigest,
		Chunker:              chunker,
		RepresentationKinds:  kinds,
		RepresentationDigest: representationDigest,
		Lexical:              lexical,
		Vector:               vector,
	})
	if err != nil {
		return "", errors.Wrap(err, "calculate bundle ID")
	}
	return id, nil
}

func representationKinds(representations []rag.Representation) []string {
	seen := make(map[string]struct{})
	for _, representation := range representations {
		seen[representation.Kind] = struct{}{}
	}
	kinds := make([]string, 0, len(seen))
	for kind := range seen {
		kinds = append(kinds, kind)
	}
	sort.Strings(kinds)
	return kinds
}

func cloneVectorIdentity(value *VectorIdentity) *VectorIdentity {
	if value == nil {
		return nil
	}
	clone := *value
	return &clone
}
