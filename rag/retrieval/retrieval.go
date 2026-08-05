package retrieval

import (
	"fmt"
	"sort"

	"github.com/go-go-golems/ragkit/rag"
)

// Target selects the identity used to collapse duplicate hits. It aliases
// the shared rag.Target vocabulary; Collapse accepts only chunk and document.
type Target = rag.Target

const (
	TargetChunk    = rag.TargetChunk
	TargetDocument = rag.TargetDocument
)

// Collapse keeps the highest-ranked hit for each target within a channel.
func Collapse(hits []rag.Hit, target Target) ([]rag.Hit, error) {
	if target != TargetChunk && target != TargetDocument {
		return nil, fmt.Errorf("unsupported collapse target %q", target)
	}
	sorted := append([]rag.Hit(nil), hits...)
	sort.SliceStable(sorted, func(left, right int) bool {
		if sorted[left].Rank == sorted[right].Rank {
			if sorted[left].ChunkID != sorted[right].ChunkID {
				return sorted[left].ChunkID < sorted[right].ChunkID
			}
			return sorted[left].RepresentationID < sorted[right].RepresentationID
		}
		return sorted[left].Rank < sorted[right].Rank
	})
	seen := map[string]bool{}
	result := make([]rag.Hit, 0, len(sorted))
	for _, hit := range sorted {
		key := hit.ChunkID
		if target == TargetDocument {
			key = hit.DocumentID
		}
		if key == "" {
			return nil, fmt.Errorf("hit has empty collapse identity")
		}
		if seen[key] {
			continue
		}
		seen[key] = true
		hit.Rank = len(result) + 1
		result = append(result, hit)
	}
	return result, nil
}

// RRFConfig controls weighted reciprocal rank fusion.
type RRFConfig struct {
	RankConstant float64            `json:"rank_constant"`
	Weights      map[string]float64 `json:"weights"`
}

// WeightedRRF fuses already-collapsed channel results by chunk identity.
func WeightedRRF(channels map[string][]rag.Hit, config RRFConfig) ([]rag.FusedHit, error) {
	if config.RankConstant <= 0 {
		return nil, fmt.Errorf("RRF rank constant must be positive")
	}
	channelNames := make([]string, 0, len(channels))
	for channel := range channels {
		channelNames = append(channelNames, channel)
	}
	sort.Strings(channelNames)
	byChunk := map[string]*rag.FusedHit{}
	for _, channel := range channelNames {
		weight := config.Weights[channel]
		if weight == 0 {
			weight = 1
		}
		for _, hit := range channels[channel] {
			if hit.ChunkID == "" {
				return nil, fmt.Errorf("channel %q contains a hit without chunk identity", channel)
			}
			entry := byChunk[hit.ChunkID]
			if entry == nil {
				entry = &rag.FusedHit{ChunkID: hit.ChunkID, DocumentID: hit.DocumentID}
				byChunk[hit.ChunkID] = entry
			}
			value := weight / (config.RankConstant + float64(hit.Rank))
			entry.Score += value
			entry.Contributions = append(entry.Contributions, rag.Contribution{
				Channel: channel,
				Rank:    hit.Rank,
				Weight:  weight,
				Value:   value,
			})
		}
	}
	result := make([]rag.FusedHit, 0, len(byChunk))
	for _, hit := range byChunk {
		result = append(result, *hit)
	}
	sort.Slice(result, func(left, right int) bool {
		if result[left].Score == result[right].Score {
			return result[left].ChunkID < result[right].ChunkID
		}
		return result[left].Score > result[right].Score
	})
	for index := range result {
		result[index].Rank = index + 1
	}
	return result, nil
}

// Hydrate resolves fused hits to source chunks and optionally truncates.
func Hydrate(hits []rag.FusedHit, chunks []rag.Chunk, limit int) ([]rag.Evidence, error) {
	chunkByID := make(map[string]rag.Chunk, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ID] = chunk
	}
	if limit < 1 || limit > len(hits) {
		limit = len(hits)
	}
	evidence := make([]rag.Evidence, 0, limit)
	for _, hit := range hits[:limit] {
		chunk, ok := chunkByID[hit.ChunkID]
		if !ok {
			return nil, fmt.Errorf("cannot hydrate missing chunk %q", hit.ChunkID)
		}
		evidence = append(evidence, rag.Evidence{
			Chunk:          chunk,
			Rank:           len(evidence) + 1,
			RetrievalScore: hit.Score,
		})
	}
	return evidence, nil
}

// FromHits converts one channel into fused hits for hydration.
func FromHits(hits []rag.Hit) []rag.FusedHit {
	result := make([]rag.FusedHit, len(hits))
	for index, hit := range hits {
		result[index] = rag.FusedHit{
			ChunkID:    hit.ChunkID,
			DocumentID: hit.DocumentID,
			Rank:       index + 1,
			Score:      hit.Score,
			Contributions: []rag.Contribution{{
				Channel: hit.Channel,
				Rank:    index + 1,
				Weight:  1,
				Value:   hit.Score,
			}},
		}
	}
	return result
}
