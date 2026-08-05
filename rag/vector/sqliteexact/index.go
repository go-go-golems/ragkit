package sqliteexact

import (
	"container/heap"
	"context"
	"database/sql"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-go-golems/ragkit/internal/fsutil"
	"github.com/go-go-golems/ragkit/rag"
	vectorutil "github.com/go-go-golems/ragkit/vector"
	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"
)

type Config struct {
	Path    string `json:"path"`
	Model   string `json:"model"`
	Channel string `json:"channel"`
}

type Manifest struct {
	Backend             string `json:"backend"`
	Version             int    `json:"version"`
	Model               string `json:"model"`
	Dimensions          int    `json:"dimensions"`
	RepresentationCount int    `json:"representation_count"`
}

// Entry is one persisted vector and its retrieval identity. It is exposed so
// backend experiments can use an immutable SQLite exact bundle as their
// correctness oracle without recomputing embeddings.
type Entry struct {
	RepresentationID string    `json:"representation_id"`
	ChunkID          string    `json:"chunk_id"`
	DocumentID       string    `json:"document_id"`
	Values           []float32 `json:"values"`
}

type Index struct {
	db       *sql.DB
	model    string
	channel  string
	embedder rag.Embedder
}

var _ rag.Index = (*Index)(nil)

func Build(ctx context.Context, cfg Config, representations []rag.Representation, chunks []rag.Chunk, vectors []rag.Vector, embedder rag.Embedder) (*Index, error) {
	if strings.TrimSpace(cfg.Path) == "" || strings.TrimSpace(cfg.Model) == "" {
		return nil, errors.New("SQLite exact path and model are required")
	}
	if embedder == nil {
		return nil, errors.New("query embedder is required")
	}
	if len(vectors) == 0 {
		return nil, errors.New("SQLite exact index requires vectors")
	}
	if cfg.Channel == "" {
		cfg.Channel = "sqlite-exact"
	}
	if err := os.MkdirAll(filepath.Dir(cfg.Path), 0o700); err != nil {
		return nil, errors.Wrap(err, "create SQLite vector parent")
	}
	temporary, err := os.CreateTemp(filepath.Dir(cfg.Path), ".vector-partial-*.sqlite")
	if err != nil {
		return nil, errors.Wrap(err, "create SQLite vector temporary file")
	}
	temporaryPath := temporary.Name()
	if err := temporary.Close(); err != nil {
		return nil, errors.Wrap(err, "close SQLite vector temporary file")
	}
	published := false
	defer func() {
		if !published {
			_ = os.Remove(temporaryPath)
		}
	}()
	db, err := sql.Open("sqlite3", temporaryPath+"?_foreign_keys=on")
	if err != nil {
		return nil, errors.Wrap(err, "open SQLite vector database")
	}
	defer func() { _ = db.Close() }()
	_, err = db.ExecContext(ctx, `
CREATE TABLE embedding (
 representation_id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL,
 document_id TEXT NOT NULL, model TEXT NOT NULL, dimensions INTEGER NOT NULL,
 values_blob BLOB NOT NULL, content_digest TEXT NOT NULL
);
CREATE INDEX embedding_model_dimensions ON embedding(model, dimensions);`)
	if err != nil {
		return nil, errors.Wrap(err, "create SQLite vector schema")
	}
	representationByID := make(map[string]rag.Representation, len(representations))
	for _, representation := range representations {
		representationByID[representation.ID] = representation
	}
	chunkByID := make(map[string]rag.Chunk, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ID] = chunk
	}
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, errors.Wrap(err, "begin SQLite vector transaction")
	}
	dimensions := 0
	for _, vector := range vectors {
		representation, ok := representationByID[vector.RepresentationID]
		if !ok {
			_ = tx.Rollback()
			return nil, errors.Errorf("vector references unknown representation %q", vector.RepresentationID)
		}
		chunk, ok := chunkByID[representation.ChunkID]
		if !ok {
			_ = tx.Rollback()
			return nil, errors.Errorf("representation references unknown chunk %q", representation.ChunkID)
		}
		if vector.Model != cfg.Model {
			_ = tx.Rollback()
			return nil, errors.Errorf("vector model %q differs from configured model %q", vector.Model, cfg.Model)
		}
		if dimensions == 0 {
			dimensions = len(vector.Values)
		}
		if len(vector.Values) == 0 || len(vector.Values) != dimensions {
			_ = tx.Rollback()
			return nil, errors.New("inconsistent vector dimensions")
		}
		blob, err := encode(vector.Values)
		if err != nil {
			_ = tx.Rollback()
			return nil, err
		}
		_, err = tx.ExecContext(ctx, `INSERT INTO embedding VALUES (?, ?, ?, ?, ?, ?, ?)`,
			representation.ID, chunk.ID, chunk.DocumentID, cfg.Model, dimensions, blob, representation.ContentDigest)
		if err != nil {
			_ = tx.Rollback()
			return nil, errors.Wrap(err, "insert SQLite vector")
		}
	}
	if err := tx.Commit(); err != nil {
		return nil, errors.Wrap(err, "commit SQLite vectors")
	}
	if err := db.Close(); err != nil {
		return nil, errors.Wrap(err, "close SQLite vector build")
	}
	if _, err := os.Stat(cfg.Path); err == nil {
		return nil, errors.Errorf("SQLite vector destination already exists: %s", cfg.Path)
	} else if !os.IsNotExist(err) {
		return nil, errors.Wrap(err, "inspect SQLite vector destination")
	}
	if err := os.Rename(temporaryPath, cfg.Path); err != nil {
		return nil, errors.Wrap(err, "publish SQLite vector database")
	}
	published = true
	if err := fsutil.SyncDirectory(filepath.Dir(cfg.Path)); err != nil {
		return nil, errors.Wrap(err, "sync SQLite vector parent")
	}
	return Open(cfg.Path, cfg.Model, cfg.Channel, embedder)
}

func Open(path, model, channel string, embedder rag.Embedder) (*Index, error) {
	if embedder == nil {
		return nil, errors.New("query embedder is required")
	}
	db, err := sql.Open("sqlite3", "file:"+path+"?mode=ro")
	if err != nil {
		return nil, errors.Wrap(err, "open SQLite vector database")
	}
	if channel == "" {
		channel = "sqlite-exact"
	}
	manifest, err := inspectDB(db)
	if err != nil {
		_ = db.Close()
		return nil, errors.Wrap(err, "inspect SQLite exact index")
	}
	if manifest.Model != model {
		_ = db.Close()
		return nil, errors.Errorf(
			"SQLite exact model %q differs from requested model %q",
			manifest.Model, model,
		)
	}
	return &Index{db: db, model: model, channel: channel, embedder: embedder}, nil
}

// Inspect returns the persisted vector identity without opening a searcher.
func Inspect(path string) (Manifest, error) {
	db, err := sql.Open("sqlite3", "file:"+path+"?mode=ro")
	if err != nil {
		return Manifest{}, errors.Wrap(err, "open SQLite exact index for inspection")
	}
	defer func() { _ = db.Close() }()
	return inspectDB(db)
}

// ReadEntries returns all persisted vectors in stable representation-ID
// order. The returned values are detached from the database connection.
func ReadEntries(ctx context.Context, path string) ([]Entry, Manifest, error) {
	db, err := sql.Open("sqlite3", "file:"+path+"?mode=ro")
	if err != nil {
		return nil, Manifest{}, errors.Wrap(err, "open SQLite exact index for reading")
	}
	defer func() { _ = db.Close() }()
	manifest, err := inspectDB(db)
	if err != nil {
		return nil, Manifest{}, errors.Wrap(err, "inspect SQLite exact entries")
	}
	rows, err := db.QueryContext(ctx, `
SELECT representation_id, chunk_id, document_id, dimensions, values_blob
FROM embedding
ORDER BY representation_id`)
	if err != nil {
		return nil, Manifest{}, errors.Wrap(err, "read SQLite exact entries")
	}
	defer func() { _ = rows.Close() }()
	entries := make([]Entry, 0, manifest.RepresentationCount)
	for rows.Next() {
		var entry Entry
		var dimensions int
		var blob []byte
		if err := rows.Scan(&entry.RepresentationID, &entry.ChunkID, &entry.DocumentID, &dimensions, &blob); err != nil {
			return nil, Manifest{}, errors.Wrap(err, "scan SQLite exact entry")
		}
		entry.Values, err = decode(blob, dimensions)
		if err != nil {
			return nil, Manifest{}, errors.Wrapf(err, "decode SQLite exact entry %q", entry.RepresentationID)
		}
		entries = append(entries, entry)
	}
	if err := rows.Err(); err != nil {
		return nil, Manifest{}, errors.Wrap(err, "iterate SQLite exact entries")
	}
	if len(entries) != manifest.RepresentationCount {
		return nil, Manifest{}, errors.Errorf(
			"read %d SQLite exact entries, expected %d",
			len(entries), manifest.RepresentationCount,
		)
	}
	return entries, manifest, nil
}

func inspectDB(db *sql.DB) (Manifest, error) {
	var manifest Manifest
	manifest.Backend = "sqlite-exact"
	manifest.Version = 1
	row := db.QueryRow(`
SELECT model, dimensions, COUNT(*)
FROM embedding
GROUP BY model, dimensions
ORDER BY model, dimensions
LIMIT 1`)
	if err := row.Scan(
		&manifest.Model, &manifest.Dimensions, &manifest.RepresentationCount,
	); err != nil {
		return Manifest{}, errors.Wrap(err, "read SQLite exact identity")
	}
	var groups int
	if err := db.QueryRow(`
SELECT COUNT(*) FROM (
 SELECT model, dimensions FROM embedding GROUP BY model, dimensions
)`).Scan(&groups); err != nil {
		return Manifest{}, errors.Wrap(err, "count SQLite exact identities")
	}
	if groups != 1 {
		return Manifest{}, errors.Errorf(
			"SQLite exact index contains %d model/dimension identities", groups,
		)
	}
	return manifest, nil
}

func (i *Index) Search(ctx context.Context, query rag.Query, limit int) ([]rag.Hit, error) {
	if i == nil || i.db == nil {
		return nil, errors.New("SQLite exact index is unavailable")
	}
	if limit < 1 {
		return nil, errors.New("search limit must be positive")
	}
	result, err := i.embedder.Embed(ctx, rag.EmbeddingRequest{Model: i.model, Texts: []string{query.Text}})
	if err != nil {
		return nil, errors.Wrap(err, "embed vector query")
	}
	if len(result.Vectors) != 1 {
		return nil, errors.Errorf("query embedder returned %d vectors", len(result.Vectors))
	}
	return i.SearchVector(ctx, result.Vectors[0], limit)
}

// SearchVector measures the persisted exact backend without including query
// embedding latency. It is primarily used as the ANN correctness oracle.
func (i *Index) SearchVector(ctx context.Context, queryVector []float32, limit int) ([]rag.Hit, error) {
	if i == nil || i.db == nil {
		return nil, errors.New("SQLite exact index is unavailable")
	}
	if limit < 1 {
		return nil, errors.New("search limit must be positive")
	}
	rows, err := i.db.QueryContext(ctx, `SELECT representation_id, chunk_id, document_id, dimensions, values_blob FROM embedding WHERE model = ?`, i.model)
	if err != nil {
		return nil, errors.Wrap(err, "read SQLite vectors")
	}
	defer func() { _ = rows.Close() }()
	best := &hitHeap{}
	heap.Init(best)
	for rows.Next() {
		var representationID, chunkID, documentID string
		var dimensions int
		var blob []byte
		if err := rows.Scan(&representationID, &chunkID, &documentID, &dimensions, &blob); err != nil {
			return nil, errors.Wrap(err, "scan SQLite vector")
		}
		vector, err := decode(blob, dimensions)
		if err != nil {
			return nil, errors.Wrapf(err, "decode vector %q", representationID)
		}
		if len(vector) != len(queryVector) {
			return nil, errors.Errorf("query dimensions %d differ from index dimensions %d", len(queryVector), len(vector))
		}
		score, err := vectorutil.Cosine(queryVector, vector)
		if err != nil {
			return nil, errors.Wrapf(err, "score vector %q", representationID)
		}
		hit := rag.Hit{RepresentationID: representationID, ChunkID: chunkID, DocumentID: documentID, Channel: i.channel, Score: score}
		if best.Len() < limit {
			heap.Push(best, hit)
		} else if better(hit, (*best)[0]) {
			heap.Pop(best)
			heap.Push(best, hit)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, errors.Wrap(err, "iterate SQLite vectors")
	}
	hits := make([]rag.Hit, best.Len())
	for position := len(hits) - 1; position >= 0; position-- {
		hits[position] = heap.Pop(best).(rag.Hit)
	}
	for position := range hits {
		hits[position].Rank = position + 1
	}
	return hits, nil
}

func (i *Index) Close() error {
	if i == nil || i.db == nil {
		return nil
	}
	return i.db.Close()
}

func encode(values []float32) ([]byte, error) {
	if err := vectorutil.ValidateFinite(values); err != nil {
		return nil, err
	}
	data := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}
	return data, nil
}

func decode(data []byte, dimensions int) ([]float32, error) {
	if dimensions < 1 || len(data) != dimensions*4 {
		return nil, errors.Errorf("invalid vector blob length %d for %d dimensions", len(data), dimensions)
	}
	values := make([]float32, dimensions)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return values, nil
}

type hitHeap []rag.Hit

func (h hitHeap) Len() int      { return len(h) }
func (h hitHeap) Swap(a, b int) { h[a], h[b] = h[b], h[a] }
func (h hitHeap) Less(a, b int) bool {
	return better(h[b], h[a])
}
func (h *hitHeap) Push(value any) { *h = append(*h, value.(rag.Hit)) }
func (h *hitHeap) Pop() any {
	old := *h
	last := old[len(old)-1]
	*h = old[:len(old)-1]
	return last
}

func better(left, right rag.Hit) bool {
	return rag.HitRanksBefore(left, right)
}
