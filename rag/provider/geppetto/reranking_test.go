package geppetto

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	geppettorerank "github.com/go-go-golems/geppetto/pkg/rerank"
	"github.com/go-go-golems/ragkit/rag"
)

type rerankProviderFixture struct {
	model    geppettorerank.Model
	response geppettorerank.Response
	err      error
	requests []geppettorerank.Request
}

func (f *rerankProviderFixture) Model() geppettorerank.Model { return f.model }

func (f *rerankProviderFixture) Rerank(_ context.Context, request geppettorerank.Request) (geppettorerank.Response, error) {
	f.requests = append(f.requests, request)
	return f.response, f.err
}

func fixtureCandidates() []rag.Evidence {
	return []rag.Evidence{
		{
			Chunk:          rag.Chunk{ID: "chunk-a", DocumentID: "doc-a", Text: "titled candidate A"},
			Rank:           7,
			RetrievalScore: 0.4,
		},
		{
			Chunk:          rag.Chunk{ID: "chunk-b", DocumentID: "doc-b", Text: "titled candidate B"},
			Rank:           8,
			RetrievalScore: 0.3,
		},
	}
}

func validRerankResponse() geppettorerank.Response {
	cost := 0.0125
	return geppettorerank.Response{
		Provider: "fixture",
		Model:    "rerank-v1",
		Results: []geppettorerank.Result{
			{DocumentID: "chunk-b", Index: 1, Score: 0.9, Rank: 1},
			{DocumentID: "chunk-a", Index: 0, Score: 0.2, Rank: 2},
		},
		Usage: &geppettorerank.Usage{InputTokens: 23, TotalTokens: 23},
		Cost:  &cost,
	}
}

func newRerankFixture(t *testing.T, response geppettorerank.Response) (*Reranker, *rerankProviderFixture) {
	t.Helper()
	provider := &rerankProviderFixture{
		model:    geppettorerank.Model{Provider: "fixture", Name: "rerank-v1"},
		response: response,
	}
	adapter, err := NewReranker(provider)
	if err != nil {
		t.Fatal(err)
	}
	return adapter, provider
}

func TestNewRerankerValidatesProviderIdentity(t *testing.T) {
	t.Parallel()

	var nilProvider geppettorerank.Provider
	for _, test := range []struct {
		name     string
		provider geppettorerank.Provider
	}{
		{name: "nil", provider: nilProvider},
		{name: "missing provider", provider: &rerankProviderFixture{model: geppettorerank.Model{Name: "model"}}},
		{name: "missing model", provider: &rerankProviderFixture{model: geppettorerank.Model{Provider: "provider"}}},
		{name: "whitespace", provider: &rerankProviderFixture{model: geppettorerank.Model{Provider: "provider", Name: "  "}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if _, err := NewReranker(test.provider); err == nil {
				t.Fatal("expected constructor error")
			}
		})
	}
}

func TestRerankerProjectsCompleteOrderingUsageAndCost(t *testing.T) {
	t.Parallel()

	adapter, provider := newRerankFixture(t, validRerankResponse())
	candidates := fixtureCandidates()
	result, err := adapter.Rerank(t.Context(), rag.RerankRequest{
		Model:      "rerank-v1",
		Query:      rag.Query{ID: "query-1", Text: "query"},
		Candidates: candidates,
		Results:    1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(provider.requests) != 1 {
		t.Fatalf("provider calls = %d, want 1", len(provider.requests))
	}
	wantRequest := geppettorerank.Request{
		Model: "rerank-v1",
		Query: "query",
		Documents: []geppettorerank.Document{
			{ID: "chunk-a", Text: "titled candidate A"},
			{ID: "chunk-b", Text: "titled candidate B"},
		},
		TopN: 2,
	}
	if !reflect.DeepEqual(provider.requests[0], wantRequest) {
		t.Fatalf("provider request = %#v, want %#v", provider.requests[0], wantRequest)
	}
	if len(result.Evidence) != 1 || result.Evidence[0].Chunk.ID != "chunk-b" {
		t.Fatalf("evidence = %#v", result.Evidence)
	}
	if result.Evidence[0].Rank != 1 || result.Evidence[0].RerankerScore == nil || *result.Evidence[0].RerankerScore != 0.9 {
		t.Fatalf("projected winner = %#v", result.Evidence[0])
	}
	if result.Evidence[0].RetrievalScore != 0.3 || result.Evidence[0].Chunk.DocumentID != "doc-b" {
		t.Fatalf("source evidence fields drifted: %#v", result.Evidence[0])
	}
	if result.Usage.InputTokens == nil || *result.Usage.InputTokens != 23 {
		t.Fatalf("input usage = %#v", result.Usage.InputTokens)
	}
	if result.Usage.CostUSD == nil || *result.Usage.CostUSD != 0.0125 {
		t.Fatalf("cost = %#v", result.Usage.CostUSD)
	}
	if candidates[1].Rank != 8 || candidates[1].RerankerScore != nil {
		t.Fatalf("caller candidates mutated: %#v", candidates)
	}
}

func TestRerankerUsesProviderDeterministicTieOrder(t *testing.T) {
	t.Parallel()

	response := validRerankResponse()
	response.Results = []geppettorerank.Result{
		{DocumentID: "chunk-a", Index: 0, Score: 0.5, Rank: 1},
		{DocumentID: "chunk-b", Index: 1, Score: 0.5, Rank: 2},
	}
	adapter, _ := newRerankFixture(t, response)
	result, err := adapter.Rerank(t.Context(), rag.RerankRequest{
		Query: rag.Query{Text: "query"}, Candidates: fixtureCandidates(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := []string{result.Evidence[0].Chunk.ID, result.Evidence[1].Chunk.ID}; !reflect.DeepEqual(got, []string{"chunk-a", "chunk-b"}) {
		t.Fatalf("tie order = %v", got)
	}
}

func TestRerankerRejectsInvalidRequestsBeforeProviderCall(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request rag.RerankRequest
	}{
		{name: "model mismatch", request: rag.RerankRequest{Model: "other", Query: rag.Query{Text: "query"}, Candidates: fixtureCandidates()}},
		{name: "no candidates", request: rag.RerankRequest{Query: rag.Query{Text: "query"}}},
		{name: "empty query", request: rag.RerankRequest{Candidates: fixtureCandidates()}},
		{name: "empty id", request: rag.RerankRequest{Query: rag.Query{Text: "query"}, Candidates: []rag.Evidence{{Chunk: rag.Chunk{Text: "text"}}}}},
		{name: "duplicate id", request: rag.RerankRequest{Query: rag.Query{Text: "query"}, Candidates: []rag.Evidence{{Chunk: rag.Chunk{ID: "same", Text: "one"}}, {Chunk: rag.Chunk{ID: " same ", Text: "two"}}}}},
		{name: "empty text", request: rag.RerankRequest{Query: rag.Query{Text: "query"}, Candidates: []rag.Evidence{{Chunk: rag.Chunk{ID: "chunk", Text: "  "}}}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			adapter, provider := newRerankFixture(t, validRerankResponse())
			if _, err := adapter.Rerank(t.Context(), test.request); err == nil {
				t.Fatal("expected request error")
			}
			if len(provider.requests) != 0 {
				t.Fatalf("provider called for invalid request: %#v", provider.requests)
			}
		})
	}
}

func TestRerankerRejectsInvalidProviderResponses(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*geppettorerank.Response)
		want   string
	}{
		{name: "provider mismatch", mutate: func(r *geppettorerank.Response) { r.Provider = "other" }, want: "provider mismatch"},
		{name: "model mismatch", mutate: func(r *geppettorerank.Response) { r.Model = "other" }, want: "model mismatch"},
		{name: "incomplete", mutate: func(r *geppettorerank.Response) { r.Results = r.Results[:1] }, want: "incomplete"},
		{name: "unknown id", mutate: func(r *geppettorerank.Response) { r.Results[0].DocumentID = "unknown" }, want: "unknown chunk ID"},
		{name: "duplicate id", mutate: func(r *geppettorerank.Response) {
			r.Results[1].DocumentID = r.Results[0].DocumentID
			r.Results[1].Index = r.Results[0].Index
		}, want: "repeats chunk ID"},
		{name: "index mismatch", mutate: func(r *geppettorerank.Response) { r.Results[0].Index = 0 }, want: "index mismatch"},
		{name: "rank mismatch", mutate: func(r *geppettorerank.Response) { r.Results[0].Rank = 2 }, want: "rank mismatch"},
		{name: "unordered", mutate: func(r *geppettorerank.Response) {
			r.Results[0], r.Results[1] = r.Results[1], r.Results[0]
			r.Results[0].Rank = 1
			r.Results[1].Rank = 2
		}, want: "not deterministically ordered"},
		{name: "nan score", mutate: func(r *geppettorerank.Response) { r.Results[0].Score = math.NaN() }, want: "non-finite"},
		{name: "infinite score", mutate: func(r *geppettorerank.Response) { r.Results[0].Score = math.Inf(1) }, want: "non-finite"},
		{name: "negative usage", mutate: func(r *geppettorerank.Response) { r.Usage.InputTokens = -1 }, want: "negative usage"},
		{name: "nan cost", mutate: func(r *geppettorerank.Response) { value := math.NaN(); r.Cost = &value }, want: "invalid cost"},
		{name: "negative cost", mutate: func(r *geppettorerank.Response) { value := -1.0; r.Cost = &value }, want: "invalid cost"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			response := validRerankResponse()
			test.mutate(&response)
			adapter, _ := newRerankFixture(t, response)
			result, err := adapter.Rerank(t.Context(), rag.RerankRequest{
				Query: rag.Query{Text: "query"}, Candidates: fixtureCandidates(),
			})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want substring %q", err, test.want)
			}
			if test.name != "negative usage" && (result.Usage.InputTokens == nil || *result.Usage.InputTokens != 23) {
				t.Fatalf("known usage lost on invalid response: %#v", result.Usage)
			}
		})
	}
}

func TestRerankerPreservesUsageAndCostOnProviderError(t *testing.T) {
	t.Parallel()

	providerErr := errors.New("fixture provider failure")
	response := validRerankResponse()
	adapter, provider := newRerankFixture(t, response)
	provider.err = providerErr
	result, err := adapter.Rerank(t.Context(), rag.RerankRequest{
		Query: rag.Query{Text: "query"}, Candidates: fixtureCandidates(),
	})
	if !errors.Is(err, providerErr) {
		t.Fatalf("error = %v, want wrapped provider error", err)
	}
	if result.Usage.InputTokens == nil || *result.Usage.InputTokens != 23 {
		t.Fatalf("input usage on error = %#v", result.Usage.InputTokens)
	}
	if result.Usage.CostUSD == nil || *result.Usage.CostUSD != 0.0125 {
		t.Fatalf("cost on error = %#v", result.Usage.CostUSD)
	}
	if len(result.Evidence) != 0 {
		t.Fatalf("partial evidence exposed on provider error: %#v", result.Evidence)
	}
}

func TestRerankerPreservesUnknownUsageAndExplicitZeroCost(t *testing.T) {
	t.Parallel()

	response := validRerankResponse()
	response.Usage = nil
	zero := 0.0
	response.Cost = &zero
	adapter, _ := newRerankFixture(t, response)
	result, err := adapter.Rerank(t.Context(), rag.RerankRequest{
		Query: rag.Query{Text: "query"}, Candidates: fixtureCandidates(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage.InputTokens != nil {
		t.Fatalf("unknown usage became known: %#v", result.Usage.InputTokens)
	}
	if result.Usage.CostUSD == nil || *result.Usage.CostUSD != 0 {
		t.Fatalf("explicit zero cost lost: %#v", result.Usage.CostUSD)
	}
}

func TestNilRerankerFailsClosed(t *testing.T) {
	t.Parallel()

	var adapter *Reranker
	if _, err := adapter.Rerank(t.Context(), rag.RerankRequest{}); err == nil {
		t.Fatal("expected nil adapter error")
	}
}
