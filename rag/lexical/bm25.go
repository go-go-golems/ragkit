package lexical

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/go-go-golems/ragkit/rag"
	textutil "github.com/go-go-golems/ragkit/text"
)

// Config controls BM25 scoring.
type Config struct {
	K1      float64  `json:"k1"`
	B       *float64 `json:"b,omitempty"`
	Channel string   `json:"channel"`
}

// BM25 is a small in-memory lexical index.
type BM25 struct {
	config          Config
	representations []rag.Representation
	chunks          map[string]rag.Chunk
	termFrequency   []map[string]int
	documentFreq    map[string]int
	lengths         []int
	averageLength   float64
}

var _ rag.Index = (*BM25)(nil)

// Build constructs an immutable in-memory BM25 index.
func Build(representations []rag.Representation, chunks []rag.Chunk, config Config) (*BM25, error) {
	if len(representations) == 0 {
		return nil, fmt.Errorf("lexical index requires representations")
	}
	if config.K1 == 0 {
		config.K1 = 1.2
	}
	if config.B == nil {
		config.B = float64Pointer(0.75)
	}
	if math.IsNaN(config.K1) || math.IsInf(config.K1, 0) ||
		math.IsNaN(*config.B) || math.IsInf(*config.B, 0) ||
		config.K1 <= 0 || *config.B < 0 || *config.B > 1 {
		return nil, fmt.Errorf("invalid BM25 parameters")
	}
	b := *config.B
	config.B = &b
	if config.Channel == "" {
		config.Channel = "bm25"
	}
	chunkByID := make(map[string]rag.Chunk, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ID] = chunk
	}
	index := &BM25{
		config:          config,
		representations: append([]rag.Representation(nil), representations...),
		chunks:          chunkByID,
		documentFreq:    map[string]int{},
	}
	var totalLength int
	representationIDs := make(map[string]struct{}, len(index.representations))
	for _, representation := range index.representations {
		if _, duplicate := representationIDs[representation.ID]; duplicate {
			return nil, fmt.Errorf("duplicate representation ID %q", representation.ID)
		}
		representationIDs[representation.ID] = struct{}{}
		if _, ok := chunkByID[representation.ChunkID]; !ok {
			return nil, fmt.Errorf("representation %q references unknown chunk %q", representation.ID, representation.ChunkID)
		}
		frequency := map[string]int{}
		terms := textutil.Terms(representation.Text)
		for _, token := range terms {
			frequency[token]++
		}
		for token := range frequency {
			index.documentFreq[token]++
		}
		index.termFrequency = append(index.termFrequency, frequency)
		index.lengths = append(index.lengths, len(terms))
		totalLength += index.lengths[len(index.lengths)-1]
	}
	index.averageLength = float64(totalLength) / float64(len(index.representations))
	return index, nil
}

// Search returns stable score-descending hits.
func (index *BM25) Search(ctx context.Context, query rag.Query, limit int) ([]rag.Hit, error) {
	if index == nil {
		return nil, fmt.Errorf("lexical index is nil")
	}
	if limit < 1 {
		return nil, fmt.Errorf("search limit must be positive")
	}
	queryTokens := textutil.Terms(query.Text)
	hits := make([]rag.Hit, 0, len(index.representations))
	total := float64(len(index.representations))
	for representationIndex, representation := range index.representations {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var score float64
		for _, token := range queryTokens {
			frequency := float64(index.termFrequency[representationIndex][token])
			if frequency == 0 {
				continue
			}
			documentFrequency := float64(index.documentFreq[token])
			idf := math.Log(1 + (total-documentFrequency+0.5)/(documentFrequency+0.5))
			lengthNorm := 1 - *index.config.B
			if index.averageLength > 0 {
				lengthNorm += *index.config.B * float64(index.lengths[representationIndex]) / index.averageLength
			}
			score += idf * frequency * (index.config.K1 + 1) / (frequency + index.config.K1*lengthNorm)
		}
		if score == 0 {
			continue
		}
		chunk := index.chunks[representation.ChunkID]
		hits = append(hits, rag.Hit{
			RepresentationID: representation.ID,
			ChunkID:          chunk.ID,
			DocumentID:       chunk.DocumentID,
			Channel:          index.config.Channel,
			Score:            score,
		})
	}
	sort.Slice(hits, func(left, right int) bool {
		return rag.HitRanksBefore(hits[left], hits[right])
	})
	if len(hits) > limit {
		hits = hits[:limit]
	}
	for position := range hits {
		hits[position].Rank = position + 1
	}
	return hits, nil
}

func (index *BM25) Close() error { return nil }

func float64Pointer(value float64) *float64 {
	return &value
}
