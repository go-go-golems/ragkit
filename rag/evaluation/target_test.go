package evaluation

import (
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func TestTargetResolver(t *testing.T) {
	t.Parallel()
	resolver := TargetResolver{Documents: map[string]rag.Document{
		"doc": {ID: "doc", Metadata: map[string]string{"evaluation_unit_id": "unit"}},
	}}
	hit := rag.Hit{RepresentationID: "rep", ChunkID: "chunk", DocumentID: "doc"}
	for target, want := range map[Target]string{
		TargetRepresentation: "rep",
		TargetChunk:          "chunk",
		TargetDocument:       "doc",
		TargetUnit:           "unit",
	} {
		got, err := resolver.HitID(hit, target)
		require.NoError(t, err)
		require.Equal(t, want, got)
	}
	_, err := resolver.HitID(hit, Target("unknown"))
	require.Error(t, err)
	_, err = (TargetResolver{}).HitID(hit, TargetUnit)
	require.ErrorContains(t, err, "unknown document")
	resolver.Documents["doc"] = rag.Document{ID: "doc"}
	_, err = resolver.HitID(hit, TargetUnit)
	require.ErrorContains(t, err, "no evaluation unit")
}

func TestTargetResolverRejectsMissingHitIdentities(t *testing.T) {
	t.Parallel()
	resolver := TargetResolver{}
	for _, target := range []Target{TargetRepresentation, TargetChunk, TargetDocument} {
		_, err := resolver.HitID(rag.Hit{}, target)
		require.Error(t, err)
	}
}
