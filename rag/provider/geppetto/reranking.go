package geppetto

import (
	"context"
	"fmt"
	"math"
	"strings"

	geppettorerank "github.com/go-go-golems/geppetto/pkg/rerank"
	"github.com/go-go-golems/ragkit/rag"
)

// Reranker adapts one already-configured Geppetto reranking provider.
type Reranker struct {
	provider geppettorerank.Provider
	model    geppettorerank.Model
}

var _ rag.Reranker = (*Reranker)(nil)

// NewReranker validates and wraps a configured Geppetto reranking provider.
func NewReranker(provider geppettorerank.Provider) (*Reranker, error) {
	if provider == nil {
		return nil, fmt.Errorf("reranking provider is required")
	}
	model := provider.Model()
	if strings.TrimSpace(model.Provider) == "" || strings.TrimSpace(model.Name) == "" {
		return nil, fmt.Errorf("reranking provider has invalid provider/model metadata")
	}
	return &Reranker{provider: provider, model: model}, nil
}

// Rerank projects ragkit evidence to Geppetto documents and validates that the
// provider returned one complete, deterministic score ordering before applying
// the caller's result limit. Provider usage and cost survive the projection,
// including when a provider returns observations together with an error.
func (r *Reranker) Rerank(ctx context.Context, request rag.RerankRequest) (rag.RerankResult, error) {
	if r == nil || r.provider == nil {
		return rag.RerankResult{}, fmt.Errorf("reranking provider is unavailable")
	}
	if request.Model != "" && request.Model != r.model.Name {
		return rag.RerankResult{}, fmt.Errorf(
			"reranking model mismatch: requested %q, provider %q",
			request.Model,
			r.model.Name,
		)
	}
	if len(request.Candidates) == 0 {
		return rag.RerankResult{}, fmt.Errorf("reranking candidates are required")
	}

	documents := make([]geppettorerank.Document, len(request.Candidates))
	candidateByID := make(map[string]rag.Evidence, len(request.Candidates))
	indexByID := make(map[string]int, len(request.Candidates))
	for i, candidate := range request.Candidates {
		id := strings.TrimSpace(candidate.Chunk.ID)
		if id == "" {
			return rag.RerankResult{}, fmt.Errorf("candidate %d has no chunk ID", i)
		}
		if _, exists := candidateByID[id]; exists {
			return rag.RerankResult{}, fmt.Errorf("duplicate candidate chunk ID %q", id)
		}
		candidateByID[id] = candidate
		indexByID[id] = i
		documents[i] = geppettorerank.Document{ID: id, Text: candidate.Chunk.Text}
	}

	providerRequest := geppettorerank.Request{
		Model:     r.model.Name,
		Query:     request.Query.Text,
		Documents: documents,
		TopN:      len(documents),
	}
	if err := geppettorerank.ValidateRequest(providerRequest, r.model); err != nil {
		return rag.RerankResult{}, fmt.Errorf("invalid Geppetto rerank request: %w", err)
	}

	response, err := r.provider.Rerank(ctx, providerRequest)
	usage, usageErr := projectRerankUsage(response)
	failWithUsage := func(failure error) (rag.RerankResult, error) {
		return rag.RerankResult{Usage: usage}, failure
	}
	if err != nil {
		return failWithUsage(fmt.Errorf("rerank with Geppetto: %w", err))
	}
	if usageErr != nil {
		return failWithUsage(usageErr)
	}
	if response.Provider != r.model.Provider {
		return failWithUsage(fmt.Errorf(
			"reranking response provider mismatch: got %q, want %q",
			response.Provider,
			r.model.Provider,
		))
	}
	if response.Model != r.model.Name {
		return failWithUsage(fmt.Errorf(
			"reranking response model mismatch: got %q, want %q",
			response.Model,
			r.model.Name,
		))
	}
	if len(response.Results) != len(documents) {
		return failWithUsage(fmt.Errorf(
			"reranking response incomplete: got %d, want %d",
			len(response.Results),
			len(documents),
		))
	}

	evidence := make([]rag.Evidence, 0, len(response.Results))
	seen := make(map[string]struct{}, len(response.Results))
	for position, result := range response.Results {
		candidate, ok := candidateByID[result.DocumentID]
		if !ok {
			return failWithUsage(fmt.Errorf(
				"reranking response contains unknown chunk ID %q",
				result.DocumentID,
			))
		}
		if _, duplicate := seen[result.DocumentID]; duplicate {
			return failWithUsage(fmt.Errorf(
				"reranking response repeats chunk ID %q",
				result.DocumentID,
			))
		}
		if result.Index != indexByID[result.DocumentID] {
			return failWithUsage(fmt.Errorf(
				"reranking response index mismatch for %q: got %d, want %d",
				result.DocumentID,
				result.Index,
				indexByID[result.DocumentID],
			))
		}
		if result.Rank != position+1 {
			return failWithUsage(fmt.Errorf(
				"reranking response rank mismatch for %q: got %d, want %d",
				result.DocumentID,
				result.Rank,
				position+1,
			))
		}
		if math.IsNaN(result.Score) || math.IsInf(result.Score, 0) {
			return failWithUsage(fmt.Errorf(
				"reranking response for %q has non-finite score",
				result.DocumentID,
			))
		}
		if position > 0 && resultRanksBefore(result, response.Results[position-1]) {
			return failWithUsage(fmt.Errorf("reranking response is not deterministically ordered"))
		}

		seen[result.DocumentID] = struct{}{}
		score := result.Score
		candidate.Rank = position + 1
		candidate.RerankerScore = &score
		evidence = append(evidence, candidate)
	}

	if request.Results > 0 && len(evidence) > request.Results {
		evidence = evidence[:request.Results]
	}
	return rag.RerankResult{Evidence: evidence, Usage: usage}, nil
}

func resultRanksBefore(left, right geppettorerank.Result) bool {
	if left.Score != right.Score {
		return left.Score > right.Score
	}
	if left.Index != right.Index {
		return left.Index < right.Index
	}
	return left.DocumentID < right.DocumentID
}

func projectRerankUsage(response geppettorerank.Response) (rag.Usage, error) {
	var usage rag.Usage
	if response.Usage != nil {
		if response.Usage.InputTokens < 0 || response.Usage.TotalTokens < 0 {
			return usage, fmt.Errorf("reranking response contains negative usage")
		}
		inputTokens := int64(response.Usage.InputTokens)
		usage.InputTokens = &inputTokens
	}
	if response.Cost != nil {
		if math.IsNaN(*response.Cost) || math.IsInf(*response.Cost, 0) || *response.Cost < 0 {
			return usage, fmt.Errorf("reranking response contains invalid cost")
		}
		cost := *response.Cost
		usage.CostUSD = &cost
	}
	return usage, nil
}
