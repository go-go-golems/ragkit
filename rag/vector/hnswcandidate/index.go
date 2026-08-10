package hnswcandidate

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"

	"github.com/coder/hnsw"
	"github.com/go-go-golems/ragkit/rag"
	vectorutil "github.com/go-go-golems/ragkit/vector"
	"github.com/pkg/errors"
)

type Config struct {
	Model          string  `json:"model"`
	Channel        string  `json:"channel"`
	M              int     `json:"m"`
	Ml             float64 `json:"ml"`
	EfConstruction int     `json:"ef_construction"`
	EfSearch       int     `json:"ef_search"`
	Seed           int64   `json:"seed"`
}

type Entry struct {
	RepresentationID string
	ChunkID          string
	DocumentID       string
	Values           []float32
}

func (c Config) Validate() error {
	if strings.TrimSpace(c.Model) == "" {
		return errors.New("HNSW model is required")
	}
	if c.M < 2 || math.IsNaN(c.Ml) || math.IsInf(c.Ml, 0) || c.Ml <= 0 || c.Ml >= 1 || c.EfConstruction < c.M || c.EfSearch < 1 {
		return errors.New("HNSW requires M >= 2, 0 < Ml < 1, ef-construction >= M, and ef-search >= 1")
	}
	return nil
}

type Index struct {
	mu         sync.Mutex
	graph      *hnsw.Graph[string]
	entries    map[string]Entry
	model      string
	channel    string
	dimensions int
	embedder   rag.Embedder
}

var _ rag.Index = (*Index)(nil)

func Build(cfg Config, entries []Entry, embedder rag.Embedder) (*Index, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if embedder == nil {
		return nil, errors.New("query embedder is required")
	}
	if len(entries) == 0 {
		return nil, errors.New("HNSW requires at least one entry")
	}
	stable := append([]Entry(nil), entries...)
	sort.Slice(stable, func(a, b int) bool { return stable[a].RepresentationID < stable[b].RepresentationID })
	dimensions := len(stable[0].Values)
	if dimensions == 0 {
		return nil, errors.New("HNSW entries require non-empty vectors")
	}
	graph := hnsw.NewGraph[string]()
	graph.M = cfg.M
	graph.Ml = cfg.Ml
	graph.EfSearch = cfg.EfConstruction
	graph.Distance = hnsw.CosineDistance
	// #nosec G404 -- reproducible HNSW construction requires a caller-owned deterministic seed.
	graph.Rng = rand.New(rand.NewSource(cfg.Seed))
	entryByID := make(map[string]Entry, len(stable))
	for _, entry := range stable {
		if strings.TrimSpace(entry.RepresentationID) == "" || strings.TrimSpace(entry.ChunkID) == "" || strings.TrimSpace(entry.DocumentID) == "" {
			return nil, errors.New("HNSW entry has an incomplete retrieval identity")
		}
		if len(entry.Values) != dimensions {
			return nil, errors.New("HNSW entries have inconsistent dimensions")
		}
		if _, exists := entryByID[entry.RepresentationID]; exists {
			return nil, errors.Errorf("duplicate HNSW representation %q", entry.RepresentationID)
		}
		if err := vectorutil.ValidateFinite(entry.Values); err != nil {
			return nil, errors.Wrapf(err, "validate HNSW representation %q", entry.RepresentationID)
		}
		entry.Values = append([]float32(nil), entry.Values...)
		entryByID[entry.RepresentationID] = entry
		graph.Add(hnsw.MakeNode(entry.RepresentationID, entry.Values))
	}
	graph.EfSearch = cfg.EfSearch
	channel := cfg.Channel
	if channel == "" {
		channel = "hnsw"
	}
	return &Index{graph: graph, entries: entryByID, model: cfg.Model, channel: channel, dimensions: dimensions, embedder: embedder}, nil
}
func (i *Index) SetEfSearch(value int) error {
	if value < 1 {
		return errors.New("ef-search must be positive")
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	i.graph.EfSearch = value
	return nil
}
func (i *Index) Search(ctx context.Context, query rag.Query, limit int) ([]rag.Hit, error) {
	if i == nil || i.graph == nil {
		return nil, errors.New("HNSW index is unavailable")
	}
	if limit < 1 {
		return nil, errors.New("search limit must be positive")
	}
	result, err := i.embedder.Embed(ctx, rag.EmbeddingRequest{Model: i.model, Texts: []string{query.Text}})
	if err != nil {
		return nil, errors.Wrap(err, "embed HNSW query")
	}
	if len(result.Vectors) != 1 {
		return nil, errors.Errorf("query embedder returned %d vectors", len(result.Vectors))
	}
	return i.SearchVector(result.Vectors[0], limit)
}
func (i *Index) SearchVector(queryVector []float32, limit int) ([]rag.Hit, error) {
	if i == nil || i.graph == nil {
		return nil, errors.New("HNSW index is unavailable")
	}
	if limit < 1 {
		return nil, errors.New("search limit must be positive")
	}
	if len(queryVector) != i.dimensions {
		return nil, errors.Errorf("HNSW query has %d dimensions, want %d", len(queryVector), i.dimensions)
	}
	if err := vectorutil.ValidateFinite(queryVector); err != nil {
		return nil, errors.Wrap(err, "validate HNSW query vector")
	}
	i.mu.Lock()
	// Search beyond the requested output limit before applying our canonical
	// score/identity order. HNSW may choose an arbitrary member at a tied
	// score boundary when asked for exactly limit nodes; overfetching to the
	// configured search breadth makes ties deterministic without changing the
	// bounded public result.
	searchLimit := max(limit, i.graph.EfSearch)
	nodes := i.graph.Search(queryVector, searchLimit)
	i.mu.Unlock()
	hits := make([]rag.Hit, 0, len(nodes))
	for _, node := range nodes {
		entry, ok := i.entries[node.Key]
		if !ok {
			return nil, errors.Errorf("HNSW returned unknown representation %q", node.Key)
		}
		score, err := vectorutil.Cosine(queryVector, node.Value)
		if err != nil {
			return nil, errors.Wrapf(err, "score HNSW representation %q", node.Key)
		}
		hits = append(hits, rag.Hit{RepresentationID: entry.RepresentationID, ChunkID: entry.ChunkID, DocumentID: entry.DocumentID, Channel: i.channel, Score: score})
	}
	sort.Slice(hits, func(a, b int) bool { return rag.HitRanksBefore(hits[a], hits[b]) })
	if len(hits) > limit {
		hits = hits[:limit]
	}
	for position := range hits {
		hits[position].Rank = position + 1
	}
	return hits, nil
}
func (i *Index) Close() error { return nil }
