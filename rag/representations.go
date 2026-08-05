package rag

import (
	"fmt"

	"github.com/go-go-golems/ragkit/digest"
)

// RawRepresentations creates one searchable raw-text representation per chunk.
func RawRepresentations(chunks []Chunk) ([]Representation, error) {
	result := make([]Representation, len(chunks))
	for i, chunk := range chunks {
		if chunk.ID == "" || chunk.ContentDigest == "" {
			return nil, fmt.Errorf("chunk %d requires ID and content digest", i)
		}
		identity, err := digest.JSON(struct {
			ChunkID       string `json:"chunk_id"`
			ContentDigest string `json:"content_digest"`
			Kind          string `json:"kind"`
		}{ChunkID: chunk.ID, ContentDigest: chunk.ContentDigest, Kind: "raw"})
		if err != nil {
			return nil, fmt.Errorf("build representation identity: %w", err)
		}
		result[i] = Representation{
			ID: "rep-" + identity[:16], ChunkID: chunk.ID, Kind: "raw",
			Text: chunk.Text, ContentDigest: chunk.ContentDigest,
		}
	}
	return result, nil
}
