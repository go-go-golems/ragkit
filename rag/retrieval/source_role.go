package retrieval

import (
	"context"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// SourceRoleSearcher restricts an existing search channel to source documents
// carrying one of the reviewed corpus source_role values.
type SourceRoleSearcher struct {
	searcher  rag.Searcher
	documents map[string]string
	allowed   map[string]struct{}
	depth     int
}

var _ rag.Searcher = (*SourceRoleSearcher)(nil)

func NewSourceRoleSearcher(searcher rag.Searcher, documents []rag.Document, searchDepth int, allowedRoles ...string) (*SourceRoleSearcher, error) {
	if searcher == nil {
		return nil, errors.New("source-role searcher requires an underlying searcher")
	}
	if len(documents) == 0 {
		return nil, errors.New("source-role searcher requires documents")
	}
	if searchDepth < 1 {
		return nil, errors.New("source-role searcher requires a positive search depth")
	}
	allowed := make(map[string]struct{}, len(allowedRoles))
	for _, role := range allowedRoles {
		role = strings.TrimSpace(role)
		if role == "" {
			return nil, errors.New("source-role searcher contains an empty allowed role")
		}
		allowed[role] = struct{}{}
	}
	if len(allowed) == 0 {
		return nil, errors.New("source-role searcher requires at least one allowed role")
	}
	documentRoles := make(map[string]string, len(documents))
	eligible := 0
	for _, document := range documents {
		id := strings.TrimSpace(document.ID)
		role := strings.TrimSpace(document.Metadata["source_role"])
		if id == "" {
			return nil, errors.New("source-role searcher received a document without an ID")
		}
		if _, duplicate := documentRoles[id]; duplicate {
			return nil, errors.Errorf("source-role searcher received duplicate document %q", id)
		}
		documentRoles[id] = role
		if _, ok := allowed[role]; ok {
			eligible++
		}
	}
	if eligible == 0 {
		return nil, errors.Errorf("source-role searcher has no documents for allowed roles %v", allowedRoles)
	}
	return &SourceRoleSearcher{searcher: searcher, documents: documentRoles, allowed: allowed, depth: searchDepth}, nil
}

func (s *SourceRoleSearcher) Search(ctx context.Context, query rag.Query, topK int) ([]rag.Hit, error) {
	if topK < 1 {
		return nil, errors.New("source-role search topK must be positive")
	}
	hits, err := s.searcher.Search(ctx, query, s.depth)
	if err != nil {
		return nil, err
	}
	filtered := make([]rag.Hit, 0, min(topK, len(hits)))
	for _, hit := range hits {
		role, ok := s.documents[hit.DocumentID]
		if !ok {
			return nil, errors.Errorf("search result references unknown document %q", hit.DocumentID)
		}
		if _, ok := s.allowed[role]; !ok {
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
