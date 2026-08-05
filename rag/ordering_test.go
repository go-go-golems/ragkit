package rag

import "testing"

func TestHitRanksBeforeUsesChunkThenRepresentationIdentity(t *testing.T) {
	t.Parallel()

	if !HitRanksBefore(
		Hit{Score: 1, ChunkID: "chunk-a", RepresentationID: "rep-z"},
		Hit{Score: 1, ChunkID: "chunk-z", RepresentationID: "rep-a"},
	) {
		t.Fatal("chunk-a must rank before chunk-z when scores tie")
	}
	if !HitRanksBefore(
		Hit{Score: 1, ChunkID: "chunk-a", RepresentationID: "rep-a"},
		Hit{Score: 1, ChunkID: "chunk-a", RepresentationID: "rep-z"},
	) {
		t.Fatal("representation identity must resolve ties within one chunk")
	}
}
