package sqliteexact

import (
	"container/heap"
	"context"
	"database/sql"
	"encoding/binary"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
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
	ContentDigest       string `json:"content_digest"`
}

// Entry is one persisted vector and its retrieval identity. It is exposed so
// backend experiments can use an immutable SQLite exact bundle as their
// correctness oracle without recomputing embeddings.
type Entry struct {
	RepresentationID string    `json:"representation_id"`
	ChunkID          string    `json:"chunk_id"`
	DocumentID       string    `json:"document_id"`
	Values           []float32 `json:"values"`
	ContentDigest    string    `json:"content_digest"`
}

type Index struct {
	db       *sql.DB
	model    string
	channel  string
	embedder rag.Embedder
}

var _ rag.Index = (*Index)(nil)

func Build(ctx context.Context, cfg Config, representations []rag.Representation, chunks []rag.Chunk, vectors []rag.Vector, embedder rag.Embedder) (*Index, error) {
	representationByID := make(map[string]rag.Representation, len(representations))
	for _, representation := range representations {
		representationByID[representation.ID] = representation
	}
	chunkByID := make(map[string]rag.Chunk, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ID] = chunk
	}
	entries := make([]Entry, 0, len(vectors))
	for _, vector := range vectors {
		representation, ok := representationByID[vector.RepresentationID]
		if !ok {
			return nil, errors.Errorf("vector references unknown representation %q", vector.RepresentationID)
		}
		chunk, ok := chunkByID[representation.ChunkID]
		if !ok {
			return nil, errors.Errorf("representation references unknown chunk %q", representation.ChunkID)
		}
		if vector.Model != cfg.Model {
			return nil, errors.Errorf("vector model %q differs from configured model %q", vector.Model, cfg.Model)
		}
		entries = append(entries, Entry{
			RepresentationID: representation.ID, ChunkID: chunk.ID, DocumentID: chunk.DocumentID,
			Values: vector.Values, ContentDigest: representation.ContentDigest,
		})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].RepresentationID < entries[j].RepresentationID })
	return BuildEntries(ctx, cfg, len(entries), func(yield func(Entry) error) error {
		for _, entry := range entries {
			if err := yield(entry); err != nil {
				return err
			}
		}
		return nil
	}, embedder)
}

// BuildEntries constructs the exact vector database from a bounded producer.
// Entries must arrive in strictly increasing representation-ID order. The
// callback is synchronous and each vector is encoded before the next is read.
func BuildEntries(ctx context.Context, cfg Config, expectedCount int, produce func(func(Entry) error) error, embedder rag.Embedder) (*Index, error) {
	if strings.TrimSpace(cfg.Path) == "" || strings.TrimSpace(cfg.Model) == "" {
		return nil, errors.New("SQLite exact path and model are required")
	}
	if embedder == nil {
		return nil, errors.New("query embedder is required")
	}
	if expectedCount < 1 || produce == nil {
		return nil, errors.New("SQLite exact entry count and producer are required")
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
	db, err := sql.Open("sqlite3", sqliteURI(temporaryPath, url.Values{"_foreign_keys": {"on"}}))
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
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, errors.Wrap(err, "begin SQLite vector transaction")
	}
	dimensions := 0
	count := 0
	lastID := ""
	err = produce(func(entry Entry) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if strings.TrimSpace(entry.RepresentationID) == "" || entry.ChunkID == "" || entry.DocumentID == "" {
			return errors.New("SQLite exact entry has incomplete identity")
		}
		if lastID != "" && entry.RepresentationID <= lastID {
			return errors.Errorf("SQLite exact entries are not in strictly increasing representation-ID order at %q", entry.RepresentationID)
		}
		if dimensions == 0 {
			dimensions = len(entry.Values)
		}
		if len(entry.Values) == 0 || len(entry.Values) != dimensions {
			return errors.New("inconsistent vector dimensions")
		}
		blob, err := encode(entry.Values)
		if err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO embedding VALUES (?, ?, ?, ?, ?, ?, ?)`,
			entry.RepresentationID, entry.ChunkID, entry.DocumentID, cfg.Model, dimensions, blob, entry.ContentDigest); err != nil {
			return errors.Wrap(err, "insert SQLite vector")
		}
		count++
		lastID = entry.RepresentationID
		return nil
	})
	if err != nil {
		_ = tx.Rollback()
		return nil, errors.Wrap(err, "consume SQLite exact entries")
	}
	if count != expectedCount {
		_ = tx.Rollback()
		return nil, errors.Errorf("received %d SQLite exact entries, expected %d", count, expectedCount)
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
	db, err := sql.Open("sqlite3", sqliteURI(path, url.Values{"mode": {"ro"}}))
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
	db, err := sql.Open("sqlite3", sqliteURI(path, url.Values{"mode": {"ro"}}))
	if err != nil {
		return Manifest{}, errors.Wrap(err, "open SQLite exact index for inspection")
	}
	defer func() { _ = db.Close() }()
	return inspectDB(db)
}

// ReadEntries returns all persisted vectors in stable representation-ID
// order. The returned values are detached from the database connection.
func ReadEntries(ctx context.Context, path string) ([]Entry, Manifest, error) {
	db, err := sql.Open("sqlite3", sqliteURI(path, url.Values{"mode": {"ro"}}))
	if err != nil {
		return nil, Manifest{}, errors.Wrap(err, "open SQLite exact index for reading")
	}
	defer func() { _ = db.Close() }()
	manifest, err := inspectDB(db)
	if err != nil {
		return nil, Manifest{}, errors.Wrap(err, "inspect SQLite exact entries")
	}
	entries, err := readEntriesDB(ctx, db, manifest.RepresentationCount)
	if err != nil {
		return nil, Manifest{}, err
	}
	return entries, manifest, nil
}

func sqliteURI(path string, parameters url.Values) string {
	return (&url.URL{Scheme: "file", Path: path, RawQuery: parameters.Encode()}).String()
}

func readEntriesDB(ctx context.Context, db *sql.DB, expected int) ([]Entry, error) {
	rows, err := db.QueryContext(ctx, `
SELECT representation_id, chunk_id, document_id, dimensions, values_blob, content_digest
FROM embedding
ORDER BY representation_id`)
	if err != nil {
		return nil, errors.Wrap(err, "read SQLite exact entries")
	}
	defer func() { _ = rows.Close() }()
	entries := make([]Entry, 0, expected)
	for rows.Next() {
		var entry Entry
		var dimensions int
		var blob []byte
		if err := rows.Scan(&entry.RepresentationID, &entry.ChunkID, &entry.DocumentID, &dimensions, &blob, &entry.ContentDigest); err != nil {
			return nil, errors.Wrap(err, "scan SQLite exact entry")
		}
		entry.Values, err = decode(blob, dimensions)
		if err != nil {
			return nil, errors.Wrapf(err, "decode SQLite exact entry %q", entry.RepresentationID)
		}
		entries = append(entries, entry)
	}
	if err := rows.Err(); err != nil {
		return nil, errors.Wrap(err, "iterate SQLite exact entries")
	}
	if len(entries) != expected {
		return nil, errors.Errorf(
			"read %d SQLite exact entries, expected %d",
			len(entries), expected,
		)
	}
	return entries, nil
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
	contentDigest, err := digestEntriesDB(context.Background(), db, manifest.RepresentationCount)
	if err != nil {
		return Manifest{}, errors.Wrap(err, "digest SQLite exact entries")
	}
	manifest.ContentDigest = contentDigest
	return manifest, nil
}

func digestEntriesDB(ctx context.Context, db *sql.DB, expected int) (string, error) {
	count := 0
	value, err := digest.JSONSequence(ctx, func(yield func(Entry) error) error {
		rows, err := db.QueryContext(ctx, `
SELECT representation_id, chunk_id, document_id, dimensions, values_blob, content_digest
FROM embedding
ORDER BY representation_id`)
		if err != nil {
			return errors.Wrap(err, "read SQLite exact entries for digest")
		}
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			var entry Entry
			var dimensions int
			var blob []byte
			if err := rows.Scan(&entry.RepresentationID, &entry.ChunkID, &entry.DocumentID, &dimensions, &blob, &entry.ContentDigest); err != nil {
				return errors.Wrap(err, "scan SQLite exact digest entry")
			}
			entry.Values, err = decode(blob, dimensions)
			if err != nil {
				return errors.Wrapf(err, "decode SQLite exact digest entry %q", entry.RepresentationID)
			}
			if err := yield(entry); err != nil {
				return err
			}
			count++
		}
		return rows.Err()
	})
	if err != nil {
		return "", err
	}
	if count != expected {
		return "", errors.Errorf("digested %d SQLite exact entries, expected %d", count, expected)
	}
	return value, nil
}

// CalculateContentDigest returns the canonical digest of the logical rows
// that Build will persist. The bundle manifest uses it to reject a replaced
// database even when model, dimensions, and row count still match.
func CalculateContentDigest(representations []rag.Representation, chunks []rag.Chunk, vectors []rag.Vector) (string, error) {
	representationByID := make(map[string]rag.Representation, len(representations))
	for _, representation := range representations {
		representationByID[representation.ID] = representation
	}
	chunkByID := make(map[string]rag.Chunk, len(chunks))
	for _, chunk := range chunks {
		chunkByID[chunk.ID] = chunk
	}
	entries := make([]Entry, 0, len(vectors))
	for _, vector := range vectors {
		representation, ok := representationByID[vector.RepresentationID]
		if !ok {
			return "", errors.Errorf("vector references unknown representation %q", vector.RepresentationID)
		}
		chunk, ok := chunkByID[representation.ChunkID]
		if !ok {
			return "", errors.Errorf("representation references unknown chunk %q", representation.ChunkID)
		}
		entries = append(entries, Entry{
			RepresentationID: representation.ID,
			ChunkID:          chunk.ID,
			DocumentID:       chunk.DocumentID,
			Values:           append([]float32(nil), vector.Values...),
			ContentDigest:    representation.ContentDigest,
		})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].RepresentationID < entries[j].RepresentationID })
	return digest.JSON(entries)
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
	if err := vectorutil.ValidateFinite(queryVector); err != nil {
		return nil, errors.Wrap(err, "validate query vector")
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
		score, err := cosineBlob(queryVector, blob, dimensions)
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
	if dimensions < 1 || len(data)%4 != 0 || dimensions != len(data)/4 {
		return nil, errors.Errorf("invalid vector blob length %d for %d dimensions", len(data), dimensions)
	}
	values := make([]float32, dimensions)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return values, nil
}

// cosineBlob scores a little-endian float32 vector directly from its SQLite
// blob. SearchVector uses this path so a corpus-sized query does not allocate
// a decoded []float32 for every row. The numerical contract matches
// vector.Cosine: float64 accumulation, finite-value rejection, and zero for a
// zero-norm vector.
func cosineBlob(query []float32, data []byte, dimensions int) (float64, error) {
	if dimensions < 1 || len(data)%4 != 0 || dimensions != len(data)/4 {
		return 0, errors.Errorf("invalid vector blob length %d for %d dimensions", len(data), dimensions)
	}
	if len(query) != dimensions {
		return 0, errors.Errorf("query dimensions %d differ from index dimensions %d", len(query), dimensions)
	}
	var dot, queryNorm, vectorNorm float64
	for index, queryValue := range query {
		vectorValue := math.Float32frombits(binary.LittleEndian.Uint32(data[index*4:]))
		if math.IsNaN(float64(vectorValue)) || math.IsInf(float64(vectorValue), 0) {
			return 0, errors.Errorf("vector contains a non-finite component at %d", index)
		}
		queryFloat := float64(queryValue)
		vectorFloat := float64(vectorValue)
		dot += queryFloat * vectorFloat
		queryNorm += queryFloat * queryFloat
		vectorNorm += vectorFloat * vectorFloat
	}
	if queryNorm == 0 || vectorNorm == 0 {
		return 0, nil
	}
	return dot / math.Sqrt(queryNorm*vectorNorm), nil
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
