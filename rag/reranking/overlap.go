package reranking

import (
	"context"
	"fmt"
	"sort"

	"github.com/go-go-golems/ragkit/rag"
	textutil "github.com/go-go-golems/ragkit/text"
)

// TermOverlap ranks evidence by distinct query-term overlap.
type TermOverlap struct{}

var _ rag.Reranker = (*TermOverlap)(nil)

func (reranker *TermOverlap) Rerank(ctx context.Context, request rag.RerankRequest) (rag.RerankResult, error) {
	if err := ctx.Err(); err != nil {
		return rag.RerankResult{}, err
	}
	if request.Results < 1 {
		return rag.RerankResult{}, fmt.Errorf("rerank result count must be positive")
	}
	queryTerms := textutil.TermSet(request.Query.Text)
	result := append([]rag.Evidence(nil), request.Candidates...)
	for index := range result {
		score := 0.0
		for term := range textutil.TermSet(result[index].Chunk.Text) {
			if _, ok := queryTerms[term]; ok {
				score++
			}
		}
		result[index].RerankerScore = &score
	}
	sort.SliceStable(result, func(left, right int) bool {
		leftScore := *result[left].RerankerScore
		rightScore := *result[right].RerankerScore
		if leftScore == rightScore {
			return result[left].Rank < result[right].Rank
		}
		return leftScore > rightScore
	})
	if len(result) > request.Results {
		result = result[:request.Results]
	}
	for index := range result {
		result[index].Rank = index + 1
	}
	return rag.RerankResult{Evidence: result}, nil
}
