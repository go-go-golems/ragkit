package retrieval

import (
	"context"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// RepresentationKindSearcher restricts an existing immutable index to a
// reviewed set of representation kinds. It asks the underlying index for its
// complete result set so that filtering cannot hide a higher-ranked eligible
// representation behind ineligible results.
type RepresentationKindSearcher struct {
	searcher rag.Searcher
	kinds    map[string]string
	allowed  map[string]struct{}
	depth    int
}

var _ rag.Searcher = (*RepresentationKindSearcher)(nil)

// NewRepresentationKindSearcher constructs a deterministic kind-filtered view
// over an existing index. Representation identity is immutable and supplied by
// the same verified index bundle as the wrapped searcher.
func NewRepresentationKindSearcher(searcher rag.Searcher, representations []rag.Representation, allowedKinds ...string) (*RepresentationKindSearcher, error) {
	if searcher == nil {
		return nil, errors.New("representation-kind searcher requires an underlying searcher")
	}
	if len(representations) == 0 {
		return nil, errors.New("representation-kind searcher requires representations")
	}
	allowed := make(map[string]struct{}, len(allowedKinds))
	for _, kind := range allowedKinds {
		kind = strings.TrimSpace(kind)
		if kind == "" {
			return nil, errors.New("representation-kind searcher contains an empty allowed kind")
		}
		allowed[kind] = struct{}{}
	}
	if len(allowed) == 0 {
		return nil, errors.New("representation-kind searcher requires at least one allowed kind")
	}
	kinds := make(map[string]string, len(representations))
	eligible := 0
	for _, representation := range representations {
		id := strings.TrimSpace(representation.ID)
		kind := strings.TrimSpace(representation.Kind)
		if id == "" || kind == "" {
			return nil, errors.New("representation-kind searcher received an incomplete representation identity")
		}
		if _, duplicate := kinds[id]; duplicate {
			return nil, errors.Errorf("representation-kind searcher received duplicate representation %q", id)
		}
		kinds[id] = kind
		if _, ok := allowed[kind]; ok {
			eligible++
		}
	}
	if eligible == 0 {
		return nil, errors.Errorf("representation-kind searcher has no representations for allowed kinds %v", allowedKinds)
	}
	return &RepresentationKindSearcher{searcher: searcher, kinds: kinds, allowed: allowed, depth: len(representations)}, nil
}

func (s *RepresentationKindSearcher) Search(ctx context.Context, query rag.Query, topK int) ([]rag.Hit, error) {
	if topK < 1 {
		return nil, errors.New("representation-kind search topK must be positive")
	}
	hits, err := s.searcher.Search(ctx, query, s.depth)
	if err != nil {
		return nil, err
	}
	filtered := make([]rag.Hit, 0, min(topK, len(hits)))
	for _, hit := range hits {
		kind, ok := s.kinds[hit.RepresentationID]
		if !ok {
			return nil, errors.Errorf("search result references unknown representation %q", hit.RepresentationID)
		}
		if _, ok := s.allowed[kind]; !ok {
			continue
		}
		hit.Rank = len(filtered) + 1
		filtered = append(filtered, hit)
		if len(filtered) == topK {
			break
		}
	}
	return filtered, nil
}
