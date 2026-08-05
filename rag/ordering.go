package rag

// HitRanksBefore defines the stable score ordering shared by retrieval
// backends. Ties are broken by document identity first, then chunk
// identity, then representation identity, so equal-scoring candidates can
// never reorder between runs regardless of map or database iteration order.
// This is the complete-ordering requirement from the CoinVault production
// RAG design (GEC-RAG-PROD-001 §4.6), strengthened from the original
// rag-ttc comparator which only compared chunk and representation IDs.
func HitRanksBefore(left, right Hit) bool {
	if left.Score != right.Score {
		return left.Score > right.Score
	}
	if left.DocumentID != right.DocumentID {
		return left.DocumentID < right.DocumentID
	}
	if left.ChunkID != right.ChunkID {
		return left.ChunkID < right.ChunkID
	}
	return left.RepresentationID < right.RepresentationID
}
