package bleve

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	blevelib "github.com/blevesearch/bleve/v2"
	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/internal/fsutil"
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
	ContentDigest       string  `json:"content_digest"`
}

const ManifestVersion = 3

// Record is the complete logical row persisted by the Bleve backend.
// BuildRecords accepts these rows in strictly increasing representation-ID
// order so it can validate uniqueness and calculate its digest without
// retaining the complete input.
type Record struct {
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

// CalculateContentDigest returns the canonical digest of the logical records
// Build will index. Bundle identity uses it before the backend is published.
func CalculateContentDigest(documents []rag.Document, chunks []rag.Chunk, representations []rag.Representation) (string, error) {
	if err := validateRepresentationIDs(representations); err != nil {
		return "", err
	}
	documentByID := make(map[string]rag.Document, len(documents))
	for _, document := range documents {
		documentByID[document.ID] = document
	}
	chunkByID := make(map[string]rag.Chunk, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ID] = chunk
	}
	records := make([]Record, 0, len(representations))
	for _, representation := range representations {
		chunk, ok := chunkByID[representation.ChunkID]
		if !ok {
			return "", errors.Errorf("representation %q references unknown chunk %q", representation.ID, representation.ChunkID)
		}
		document, ok := documentByID[chunk.DocumentID]
		if !ok {
			return "", errors.Errorf("chunk %q references unknown document %q", chunk.ID, chunk.DocumentID)
		}
		records = append(records, Record{
			RepresentationID: representation.ID,
			ChunkID:          chunk.ID,
			DocumentID:       document.ID,
			Kind:             representation.Kind,
			Title:            document.Title,
			Body:             representation.Text,
		})
	}
	sort.Slice(records, func(i, j int) bool { return records[i].RepresentationID < records[j].RepresentationID })
	return digest.JSON(records)
}

func Build(ctx context.Context, cfg Config, documents []rag.Document, chunks []rag.Chunk, representations []rag.Representation) (*Index, Manifest, error) {
	if err := validateRepresentationIDs(representations); err != nil {
		return nil, Manifest{}, err
	}
	documentByID := make(map[string]rag.Document, len(documents))
	for _, document := range documents {
		documentByID[document.ID] = document
	}
	chunkByID := make(map[string]rag.Chunk, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ID] = chunk
	}
	records := make([]Record, 0, len(representations))
	for _, representation := range representations {
		chunk, ok := chunkByID[representation.ChunkID]
		if !ok {
			return nil, Manifest{}, errors.Errorf("representation %q references unknown chunk %q", representation.ID, representation.ChunkID)
		}
		document, ok := documentByID[chunk.DocumentID]
		if !ok {
			return nil, Manifest{}, errors.Errorf("chunk %q references unknown document %q", chunk.ID, chunk.DocumentID)
		}
		records = append(records, Record{
			RepresentationID: representation.ID,
			ChunkID:          chunk.ID,
			DocumentID:       document.ID,
			Kind:             representation.Kind,
			Title:            document.Title,
			Body:             representation.Text,
		})
	}
	sort.Slice(records, func(i, j int) bool { return records[i].RepresentationID < records[j].RepresentationID })
	return BuildRecords(ctx, cfg, len(records), func(yield func(Record) error) error {
		for _, current := range records {
			if err := yield(current); err != nil {
				return err
			}
		}
		return nil
	})
}

// BuildRecords constructs a Bleve index from a bounded producer. Records must
// arrive in strictly increasing representation-ID order. The callback is
// synchronous and BuildRecords retains at most one configured Bleve batch.
func BuildRecords(ctx context.Context, cfg Config, expectedCount int, produce func(func(Record) error) error) (*Index, Manifest, error) {
	cfg = defaults(cfg)
	if math.IsNaN(cfg.TitleBoost) || math.IsInf(cfg.TitleBoost, 0) || cfg.TitleBoost <= 0 ||
		math.IsNaN(cfg.BodyBoost) || math.IsInf(cfg.BodyBoost, 0) || cfg.BodyBoost <= 0 {
		return nil, Manifest{}, errors.New("Bleve title and body boosts must be finite and greater than zero")
	}
	if strings.TrimSpace(cfg.Path) == "" {
		return nil, Manifest{}, errors.New("Bleve index path is required")
	}
	if expectedCount < 1 || produce == nil {
		return nil, Manifest{}, errors.New("Bleve record count and producer are required")
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
	storedID.Index = true
	documentMapping.AddFieldMappingsAt("representation_id", storedID)
	documentMapping.AddFieldMappingsAt("chunk_id", storedID)
	documentMapping.AddFieldMappingsAt("document_id", storedID)
	documentMapping.AddFieldMappingsAt("kind", storedID)
	storedText := blevelib.NewTextFieldMapping()
	storedText.Store = true
	documentMapping.AddFieldMappingsAt("title", storedText)
	documentMapping.AddFieldMappingsAt("body", storedText)
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
	count := 0
	lastID := ""
	contentDigest, err := digest.JSONSequence(ctx, func(yieldDigest func(Record) error) error {
		return produce(func(current Record) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if strings.TrimSpace(current.RepresentationID) == "" || current.ChunkID == "" || current.DocumentID == "" {
				return errors.New("Bleve record has incomplete identity")
			}
			if lastID != "" && current.RepresentationID <= lastID {
				return errors.Errorf("Bleve records are not in strictly increasing representation-ID order at %q", current.RepresentationID)
			}
			if err := batch.Index(current.RepresentationID, current); err != nil {
				return errors.Wrap(err, "add Bleve batch record")
			}
			count++
			lastID = current.RepresentationID
			if count%cfg.BatchSize == 0 {
				if err := index.Batch(batch); err != nil {
					return errors.Wrap(err, "commit Bleve batch")
				}
				batch = index.NewBatch()
			}
			return yieldDigest(current)
		})
	})
	if err != nil {
		return nil, Manifest{}, errors.Wrap(err, "consume Bleve records")
	}
	if count != expectedCount {
		return nil, Manifest{}, errors.Errorf("received %d Bleve records, expected %d", count, expectedCount)
	}
	if batch.Size() > 0 {
		if err := index.Batch(batch); err != nil {
			return nil, Manifest{}, errors.Wrap(err, "commit final Bleve batch")
		}
	}
	manifest := Manifest{
		Backend: "bleve-bm25", Version: ManifestVersion, RepresentationCount: count,
		Channel: cfg.Channel, TitleBoost: cfg.TitleBoost, BodyBoost: cfg.BodyBoost,
		ContentDigest: contentDigest,
	}
	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return nil, Manifest{}, errors.Wrap(err, "marshal Bleve manifest")
	}
	manifestPath := filepath.Join(temporary, "rag-manifest.json")
	manifestFile, err := os.OpenFile(manifestPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, Manifest{}, errors.Wrap(err, "open Bleve manifest")
	}
	if _, err := manifestFile.Write(append(manifestData, '\n')); err != nil {
		_ = manifestFile.Close()
		return nil, Manifest{}, errors.Wrap(err, "write Bleve manifest")
	}
	if err := manifestFile.Sync(); err != nil {
		_ = manifestFile.Close()
		return nil, Manifest{}, errors.Wrap(err, "sync Bleve manifest")
	}
	if err := manifestFile.Close(); err != nil {
		return nil, Manifest{}, errors.Wrap(err, "close Bleve manifest")
	}
	if err := fsutil.SyncDirectory(temporary); err != nil {
		return nil, Manifest{}, errors.Wrap(err, "sync Bleve manifest directory")
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

func validateRepresentationIDs(representations []rag.Representation) error {
	seen := make(map[string]bool, len(representations))
	for index, representation := range representations {
		if strings.TrimSpace(representation.ID) == "" {
			return errors.Errorf("representation %d has no ID", index)
		}
		if seen[representation.ID] {
			return errors.Errorf("duplicate representation ID %q", representation.ID)
		}
		seen[representation.ID] = true
	}
	return nil
}

// InspectContentDigest reconstructs the canonical logical records from the
// persisted index. It binds a bundle to indexed content, not merely to the
// backend's record count and configuration manifest.
func InspectContentDigest(path string) (string, error) {
	data, err := os.ReadFile(filepath.Join(path, "rag-manifest.json"))
	if err != nil {
		return "", errors.Wrap(err, "read Bleve manifest")
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return "", errors.Wrap(err, "decode Bleve manifest")
	}
	index, err := blevelib.OpenUsing(path, map[string]interface{}{
		"bolt_timeout": "3s",
		"read_only":    true,
	})
	if err != nil {
		return "", errors.Wrap(err, "open Bleve index for content inspection")
	}
	defer func() { _ = index.Close() }()
	request := blevelib.NewSearchRequestOptions(blevelib.NewMatchAllQuery(), manifest.RepresentationCount, 0, false)
	request.SortBy([]string{"_id"})
	request.Fields = []string{"representation_id", "chunk_id", "document_id", "kind", "title", "body"}
	result, err := index.Search(request)
	if err != nil {
		return "", errors.Wrap(err, "read Bleve records for content inspection")
	}
	if len(result.Hits) != manifest.RepresentationCount {
		return "", errors.Errorf("Bleve index contains %d records, expected %d", len(result.Hits), manifest.RepresentationCount)
	}
	records := make([]Record, 0, len(result.Hits))
	for _, hit := range result.Hits {
		records = append(records, Record{
			RepresentationID: stringField(hit.Fields, "representation_id", hit.ID),
			ChunkID:          stringField(hit.Fields, "chunk_id", ""),
			DocumentID:       stringField(hit.Fields, "document_id", ""),
			Kind:             stringField(hit.Fields, "kind", ""),
			Title:            stringField(hit.Fields, "title", ""),
			Body:             stringField(hit.Fields, "body", ""),
		})
	}
	return digest.JSON(records)
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
	search.SortBy([]string{"-_score", "document_id", "chunk_id", "_id"})
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
