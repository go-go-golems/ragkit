package bleve

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	blevelib "github.com/blevesearch/bleve/v2"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

type Config struct {
	Path       string  `json:"path"`
	Channel    string  `json:"channel"`
	TitleBoost float64 `json:"title_boost"`
	BodyBoost  float64 `json:"body_boost"`
	BatchSize  int     `json:"batch_size"`
}

type Manifest struct {
	Backend             string  `json:"backend"`
	Version             int     `json:"version"`
	RepresentationCount int     `json:"representation_count"`
	Channel             string  `json:"channel"`
	TitleBoost          float64 `json:"title_boost"`
	BodyBoost           float64 `json:"body_boost"`
}

const ManifestVersion = 2

type record struct {
	RepresentationID string `json:"representation_id"`
	ChunkID          string `json:"chunk_id"`
	DocumentID       string `json:"document_id"`
	Kind             string `json:"kind"`
	Title            string `json:"title"`
	Body             string `json:"body"`
}

type Index struct {
	index      blevelib.Index
	channel    string
	titleBoost float64
	bodyBoost  float64
	manifest   Manifest
}

var _ rag.Index = (*Index)(nil)

func Build(ctx context.Context, cfg Config, documents []rag.Document, chunks []rag.Chunk, representations []rag.Representation) (*Index, Manifest, error) {
	cfg = defaults(cfg)
	if strings.TrimSpace(cfg.Path) == "" {
		return nil, Manifest{}, errors.New("Bleve index path is required")
	}
	if len(representations) == 0 {
		return nil, Manifest{}, errors.New("Bleve index requires representations")
	}
	documentByID := make(map[string]rag.Document, len(documents))
	for _, document := range documents {
		documentByID[document.ID] = document
	}
	chunkByID := make(map[string]rag.Chunk, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ID] = chunk
	}
	parent := filepath.Dir(cfg.Path)
	if err := os.MkdirAll(parent, 0o700); err != nil {
		return nil, Manifest{}, errors.Wrap(err, "create Bleve parent directory")
	}
	temporary, err := os.MkdirTemp(parent, ".bleve-partial-*")
	if err != nil {
		return nil, Manifest{}, errors.Wrap(err, "create Bleve temporary directory")
	}
	published := false
	defer func() {
		if !published {
			_ = os.RemoveAll(temporary)
		}
	}()
	mapping := blevelib.NewIndexMapping()
	documentMapping := blevelib.NewDocumentMapping()
	storedID := blevelib.NewKeywordFieldMapping()
	storedID.Store = true
	// Stable identities must be sortable before Bleve applies the result
	// cutoff. Keyword indexing retains exact values and document values.
	storedID.Index = true
	documentMapping.AddFieldMappingsAt("representation_id", storedID)
	documentMapping.AddFieldMappingsAt("chunk_id", storedID)
	documentMapping.AddFieldMappingsAt("document_id", storedID)
	mapping.DefaultMapping = documentMapping
	index, err := blevelib.New(temporary, mapping)
	if err != nil {
		return nil, Manifest{}, errors.Wrap(err, "create Bleve index")
	}
	closeOnError := true
	defer func() {
		if closeOnError {
			_ = index.Close()
		}
	}()
	batch := index.NewBatch()
	for i, representation := range representations {
		if err := ctx.Err(); err != nil {
			return nil, Manifest{}, err
		}
		chunk, ok := chunkByID[representation.ChunkID]
		if !ok {
			return nil, Manifest{}, errors.Errorf("representation %q references unknown chunk %q", representation.ID, representation.ChunkID)
		}
		document, ok := documentByID[chunk.DocumentID]
		if !ok {
			return nil, Manifest{}, errors.Errorf("chunk %q references unknown document %q", chunk.ID, chunk.DocumentID)
		}
		err := batch.Index(representation.ID, record{
			RepresentationID: representation.ID,
			ChunkID:          chunk.ID,
			DocumentID:       document.ID,
			Kind:             representation.Kind,
			Title:            document.Title,
			Body:             representation.Text,
		})
		if err != nil {
			return nil, Manifest{}, errors.Wrap(err, "add Bleve batch record")
		}
		if (i+1)%cfg.BatchSize == 0 {
			if err := index.Batch(batch); err != nil {
				return nil, Manifest{}, errors.Wrap(err, "commit Bleve batch")
			}
			batch = index.NewBatch()
		}
	}
	if batch.Size() > 0 {
		if err := index.Batch(batch); err != nil {
			return nil, Manifest{}, errors.Wrap(err, "commit final Bleve batch")
		}
	}
	manifest := Manifest{
		Backend: "bleve-bm25", Version: ManifestVersion, RepresentationCount: len(representations),
		Channel: cfg.Channel, TitleBoost: cfg.TitleBoost, BodyBoost: cfg.BodyBoost,
	}
	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return nil, Manifest{}, errors.Wrap(err, "marshal Bleve manifest")
	}
	if err := os.WriteFile(filepath.Join(temporary, "rag-manifest.json"), append(manifestData, '\n'), 0o600); err != nil {
		return nil, Manifest{}, errors.Wrap(err, "write Bleve manifest")
	}
	if err := index.Close(); err != nil {
		return nil, Manifest{}, errors.Wrap(err, "close built Bleve index")
	}
	closeOnError = false
	if _, err := os.Stat(cfg.Path); err == nil {
		return nil, Manifest{}, errors.Errorf("Bleve destination already exists: %s", cfg.Path)
	} else if !os.IsNotExist(err) {
		return nil, Manifest{}, errors.Wrap(err, "inspect Bleve destination")
	}
	if err := os.Rename(temporary, cfg.Path); err != nil {
		return nil, Manifest{}, errors.Wrap(err, "publish Bleve index")
	}
	published = true
	opened, err := Open(cfg.Path, cfg.Channel)
	if err != nil {
		return nil, Manifest{}, err
	}
	return opened, manifest, nil
}

func Open(path, channel string) (*Index, error) {
	// The scorch root.bolt takes an exclusive flock, and bbolt waits for it
	// forever by default. A second process opening the same bundle therefore
	// hung silently before any UI appeared. The bounded timeout turns the
	// held lock into the error below instead.
	index, err := blevelib.OpenUsing(path, map[string]interface{}{
		"bolt_timeout": "3s",
	})
	if err != nil {
		if strings.Contains(err.Error(), "timeout") {
			return nil, errors.Errorf(
				"open Bleve index: another rag-ttc process holds the lock on %s; close the other chat or corpus command using this bundle and retry",
				path,
			)
		}
		return nil, errors.Wrap(err, "open Bleve index")
	}
	data, err := os.ReadFile(filepath.Join(path, "rag-manifest.json"))
	if err != nil {
		_ = index.Close()
		return nil, errors.Wrap(err, "read Bleve manifest")
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		_ = index.Close()
		return nil, errors.Wrap(err, "decode Bleve manifest")
	}
	if manifest.Version != ManifestVersion {
		_ = index.Close()
		return nil, errors.Errorf("Bleve index version %d is unsupported; rebuild with version %d for deterministic chunk ordering", manifest.Version, ManifestVersion)
	}
	if channel == "" {
		channel = manifest.Channel
	}
	if channel == "" {
		channel = "bleve-bm25"
	}
	titleBoost, bodyBoost := manifest.TitleBoost, manifest.BodyBoost
	if titleBoost == 0 {
		titleBoost = 2
	}
	if bodyBoost == 0 {
		bodyBoost = 1
	}
	return &Index{index: index, channel: channel, titleBoost: titleBoost, bodyBoost: bodyBoost, manifest: manifest}, nil
}

func (i *Index) Search(ctx context.Context, query rag.Query, limit int) ([]rag.Hit, error) {
	if i == nil || i.index == nil {
		return nil, errors.New("Bleve index is unavailable")
	}
	if limit < 1 {
		return nil, errors.New("search limit must be positive")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	body := blevelib.NewMatchQuery(query.Text)
	body.SetField("body")
	body.SetBoost(i.bodyBoost)
	title := blevelib.NewMatchQuery(query.Text)
	title.SetField("title")
	title.SetBoost(i.titleBoost)
	search := blevelib.NewSearchRequestOptions(blevelib.NewDisjunctionQuery(title, body), limit, 0, false)
	// Bleve applies limit while executing the request. Declare the complete
	// ordering here so equal-score candidates are resolved before that cutoff;
	// sorting only the returned slice cannot recover a tied hit that Bleve did
	// not select. Build indexes representations under their stable IDs, so _id
	// is the same identity used by the defensive normalization below.
	search.SortBy([]string{"-_score", "chunk_id", "_id"})
	search.Fields = []string{"representation_id", "chunk_id", "document_id"}
	result, err := i.index.SearchInContext(ctx, search)
	if err != nil {
		return nil, errors.Wrap(err, "search Bleve index")
	}
	hits := make([]rag.Hit, 0, len(result.Hits))
	for _, hit := range result.Hits {
		hits = append(hits, rag.Hit{
			RepresentationID: stringField(hit.Fields, "representation_id", hit.ID),
			ChunkID:          stringField(hit.Fields, "chunk_id", ""),
			DocumentID:       stringField(hit.Fields, "document_id", ""),
			Channel:          i.channel,
			Score:            hit.Score,
		})
	}
	sort.SliceStable(hits, func(a, b int) bool {
		return rag.HitRanksBefore(hits[a], hits[b])
	})
	for position := range hits {
		hits[position].Rank = position + 1
	}
	return hits, nil
}

func (i *Index) Close() error {
	if i == nil || i.index == nil {
		return nil
	}
	return i.index.Close()
}

func defaults(cfg Config) Config {
	if cfg.Channel == "" {
		cfg.Channel = "bleve-bm25"
	}
	if cfg.BatchSize < 1 {
		cfg.BatchSize = 500
	}
	if cfg.TitleBoost == 0 {
		cfg.TitleBoost = 2
	}
	if cfg.BodyBoost == 0 {
		cfg.BodyBoost = 1
	}
	return cfg
}

func stringField(fields map[string]interface{}, key, fallback string) string {
	value, ok := fields[key]
	if !ok {
		return fallback
	}
	text, ok := value.(string)
	if !ok {
		return fallback
	}
	return text
}
