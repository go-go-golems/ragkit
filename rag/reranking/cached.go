package reranking

import (
	"context"
	"strings"
	"sync"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/execution"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

const (
	rerankingCacheStep    = "reranking"
	rerankingCacheVersion = "v1"
)

// CacheKeyInput covers every semantic input that can affect a reranking result.
type CacheKeyInput struct {
	Model          string                 `json:"model"`
	QueryDigest    string                 `json:"query_digest"`
	Candidates     []rag.EvidenceIdentity `json:"candidates"`
	ResultCount    int                    `json:"result_count"`
	AdapterVersion string                 `json:"adapter_version"`
}

type cacheEnvelope struct {
	Result rag.RerankResult `json:"result"`
}

type CachedOptions struct {
	Cache          execution.Cache
	Limiter        execution.Limiter
	Workers        int
	AdapterVersion string
}

type CachedReport struct {
	Cache execution.CacheReport `json:"cache"`
	Usage rag.Usage             `json:"usage"`
}

// CachedReranker decorates a Reranker with per-request durable caching,
// limiting, and cumulative cache/usage reporting.
type CachedReranker struct {
	reranker rag.Reranker
	options  CachedOptions
	mutex    sync.Mutex
	report   CachedReport
}

var _ rag.Reranker = (*CachedReranker)(nil)

// NewCachedReranker validates and constructs a cached Reranker decorator.
func NewCachedReranker(
	reranker rag.Reranker,
	options CachedOptions,
) (*CachedReranker, error) {
	if reranker == nil {
		return nil, errors.New("reranker is required")
	}
	if err := validateCachedOptions(options); err != nil {
		return nil, err
	}
	return &CachedReranker{reranker: reranker, options: options}, nil
}

// Rerank satisfies rag.Reranker and accumulates reports across calls.
func (r *CachedReranker) Rerank(
	ctx context.Context,
	request rag.RerankRequest,
) (rag.RerankResult, error) {
	if r == nil {
		return rag.RerankResult{}, errors.New("cached reranker is required")
	}
	results, report, err := Cached(ctx, []rag.RerankRequest{request}, r.options, r.reranker)
	r.mutex.Lock()
	r.report.Cache.Hits += report.Cache.Hits
	r.report.Cache.Misses += report.Cache.Misses
	r.report.Cache.Writes += report.Cache.Writes
	r.report.Cache.WorkCalls += report.Cache.WorkCalls
	r.report.Cache.Outcomes = append(r.report.Cache.Outcomes, report.Cache.Outcomes...)
	r.report.Usage.Add(report.Usage)
	r.mutex.Unlock()
	if err != nil {
		return rag.RerankResult{}, err
	}
	return results[0], nil
}

// Snapshot returns a copy of the cumulative decorator report.
func (r *CachedReranker) Snapshot() CachedReport {
	if r == nil {
		return CachedReport{}
	}
	r.mutex.Lock()
	defer r.mutex.Unlock()
	report := r.report
	report.Cache.Outcomes = append([]execution.CacheOutcome(nil), report.Cache.Outcomes...)
	return report
}

func NewCacheKey(request rag.RerankRequest, adapterVersion string) (execution.Key, error) {
	if strings.TrimSpace(adapterVersion) == "" {
		return execution.Key{}, errors.New("reranking adapter version is required")
	}
	queryDigest, err := digest.JSON(request.Query.Text)
	if err != nil {
		return execution.Key{}, errors.Wrap(err, "digest reranking query")
	}
	candidates, err := rag.EvidenceIdentities(request.Candidates)
	if err != nil {
		return execution.Key{}, err
	}
	return execution.NewKey(rerankingCacheStep, rerankingCacheVersion, CacheKeyInput{
		Model: request.Model, QueryDigest: queryDigest, Candidates: candidates,
		ResultCount: request.Results, AdapterVersion: adapterVersion,
	})
}

func Cached(
	ctx context.Context,
	requests []rag.RerankRequest,
	options CachedOptions,
	reranker rag.Reranker,
) ([]rag.RerankResult, CachedReport, error) {
	report := CachedReport{}
	if reranker == nil {
		return nil, report, errors.New("reranker is required")
	}
	if err := validateCachedOptions(options); err != nil {
		return nil, report, err
	}
	var usage rag.UsageAccumulator
	envelopes, cacheReport, err := execution.MapCached(
		ctx,
		requests,
		execution.CachedMapOptions[rag.RerankRequest]{
			Map: execution.MapOptions[rag.RerankRequest]{
				Workers: options.Workers,
				Limiter: options.Limiter,
			},
			Cache: options.Cache,
			Key: func(request rag.RerankRequest) (execution.Key, error) {
				return NewCacheKey(request, options.AdapterVersion)
			},
		},
		func(ctx context.Context, request rag.RerankRequest) (cacheEnvelope, error) {
			result, err := reranker.Rerank(ctx, request)
			if err != nil {
				return cacheEnvelope{}, err
			}
			usage.Add(result.Usage)
			return cacheEnvelope{Result: result}, nil
		},
	)
	report.Cache = cacheReport
	report.Usage = usage.Snapshot()
	if err != nil {
		return nil, report, err
	}
	results := make([]rag.RerankResult, len(envelopes))
	for i, envelope := range envelopes {
		results[i] = envelope.Result
		results[i].Usage = rag.Usage{}
	}
	return results, report, nil
}

func validateCachedOptions(options CachedOptions) error {
	if options.Cache == nil {
		return errors.New("provider cache is required")
	}
	if options.Workers < 1 {
		return errors.New("provider cache workers must be positive")
	}
	if strings.TrimSpace(options.AdapterVersion) == "" {
		return errors.New("provider adapter version is required")
	}
	return nil
}
