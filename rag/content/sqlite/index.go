// Package sqlite implements the immutable read-only content store used by a
// serving bundle. Build consumes ordered producers and never materializes the
// complete corpus in memory.
package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/internal/fsutil"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/go-go-golems/ragkit/rag/content"
	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"
)

const (
	Backend         = "sqlite-content"
	ManifestVersion = 1
	DefaultMaxBatch = 256
)

// Identity binds the logical content rows to a bundle manifest. The digest is
// over canonical document and chunk identities, never SQLite page bytes.
type Identity struct {
	Backend        string `json:"backend"`
	Version        int    `json:"version"`
	DocumentCount  int    `json:"document_count"`
	ChunkCount     int    `json:"chunk_count"`
	DocumentDigest string `json:"document_digest"`
	ChunkDigest    string `json:"chunk_digest"`
	ContentDigest  string `json:"content_digest"`
}

// Producer emits values synchronously. It must call yield in canonical source
// order and stop when yield returns an error.
type Producer[T any] func(context.Context, func(T) error) error

type BuildInput struct {
	Path      string
	Documents Producer[rag.Document]
	Chunks    Producer[rag.Chunk]
}

type BuildResult struct {
	Path     string
	Identity Identity
}

type Config struct {
	Path     string
	MaxBatch int
}

type Index struct {
	db       *sql.DB
	path     string
	maxBatch int
}

var _ content.Store = (*Index)(nil)

// Build writes a content database to a temporary sibling and atomically
// renames it into place only after both producers and integrity checks pass.
func Build(ctx context.Context, input BuildInput) (BuildResult, error) {
	if strings.TrimSpace(input.Path) == "" || input.Documents == nil || input.Chunks == nil {
		return BuildResult{}, errors.New("content path and document/chunk producers are required")
	}
	if err := os.MkdirAll(filepath.Dir(input.Path), 0o700); err != nil {
		return BuildResult{}, errors.Wrap(err, "create content parent")
	}
	temporary, err := os.CreateTemp(filepath.Dir(input.Path), ".content-partial-*.sqlite")
	if err != nil {
		return BuildResult{}, errors.Wrap(err, "create content temporary file")
	}
	temporaryPath := temporary.Name()
	if err := temporary.Close(); err != nil {
		return BuildResult{}, errors.Wrap(err, "close content temporary file")
	}
	published := false
	defer func() {
		if !published {
			_ = os.Remove(temporaryPath)
		}
	}()

	db, err := openDatabase(ctx, temporaryPath, false)
	if err != nil {
		return BuildResult{}, err
	}
	closeDB := true
	defer func() {
		if closeDB {
			_ = db.Close()
		}
	}()
	if err := createSchema(ctx, db); err != nil {
		return BuildResult{}, err
	}

	documentDigest, documentCount, err := writeDocuments(ctx, db, input.Documents)
	if err != nil {
		return BuildResult{}, errors.Wrap(err, "write content documents")
	}
	chunkDigest, chunkCount, err := writeChunks(ctx, db, input.Chunks)
	if err != nil {
		return BuildResult{}, errors.Wrap(err, "write content chunks")
	}
	identity, err := NewIdentity(documentCount, chunkCount, documentDigest, chunkDigest)
	if err != nil {
		return BuildResult{}, err
	}
	if err := writeIdentity(ctx, db, identity); err != nil {
		return BuildResult{}, err
	}
	if err := db.Close(); err != nil {
		return BuildResult{}, errors.Wrap(err, "close built content database")
	}
	closeDB = false
	if _, err := os.Stat(input.Path); err == nil {
		return BuildResult{}, errors.Errorf("content destination already exists: %s", input.Path)
	} else if !os.IsNotExist(err) {
		return BuildResult{}, errors.Wrap(err, "inspect content destination")
	}
	if err := os.Rename(temporaryPath, input.Path); err != nil {
		return BuildResult{}, errors.Wrap(err, "publish content database")
	}
	published = true
	if err := fsutil.SyncDirectory(filepath.Dir(input.Path)); err != nil {
		return BuildResult{}, errors.Wrap(err, "sync content parent")
	}
	return BuildResult{Path: input.Path, Identity: identity}, nil
}

func NewIdentity(documentCount, chunkCount int, documentDigest, chunkDigest string) (Identity, error) {
	if documentCount < 1 || chunkCount < 1 || documentDigest == "" || chunkDigest == "" {
		return Identity{}, errors.New("content identity requires positive counts and digests")
	}
	contentDigest, err := digest.TruncatedJSON("ct-", 16, struct {
		DocumentDigest string `json:"document_digest"`
		ChunkDigest    string `json:"chunk_digest"`
	}{documentDigest, chunkDigest})
	if err != nil {
		return Identity{}, errors.Wrap(err, "calculate content digest")
	}
	return Identity{
		Backend: Backend, Version: ManifestVersion,
		DocumentCount: documentCount, ChunkCount: chunkCount,
		DocumentDigest: documentDigest, ChunkDigest: chunkDigest,
		ContentDigest: contentDigest,
	}, nil
}

func createSchema(ctx context.Context, db *sql.DB) error {
	_, err := db.ExecContext(ctx, `
CREATE TABLE document (
 id TEXT PRIMARY KEY,
 ordinal INTEGER NOT NULL UNIQUE,
 source_uri TEXT NOT NULL,
 title TEXT NOT NULL,
 text TEXT NOT NULL,
 content_digest TEXT NOT NULL,
 metadata_json TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE chunk (
 id TEXT PRIMARY KEY,
 ordinal INTEGER NOT NULL UNIQUE,
 document_id TEXT NOT NULL REFERENCES document(id),
 document_ordinal INTEGER NOT NULL,
 byte_start INTEGER NOT NULL,
 byte_end INTEGER NOT NULL,
 text TEXT NOT NULL,
 content_digest TEXT NOT NULL,
 chunker TEXT NOT NULL,
 UNIQUE(document_id, document_ordinal)
) WITHOUT ROWID;
CREATE INDEX chunk_document ON chunk(document_id);`)
	if err != nil {
		return errors.Wrap(err, "create content schema")
	}
	return nil
}

func writeIdentity(ctx context.Context, db *sql.DB, identity Identity) error {
	if _, err := db.ExecContext(ctx, `
CREATE TABLE content_identity (
 id INTEGER PRIMARY KEY CHECK (id = 1),
 backend TEXT NOT NULL,
 version INTEGER NOT NULL,
 document_count INTEGER NOT NULL,
 chunk_count INTEGER NOT NULL,
 document_digest TEXT NOT NULL,
 chunk_digest TEXT NOT NULL,
 content_digest TEXT NOT NULL
) WITHOUT ROWID;
INSERT INTO content_identity VALUES (1, ?, ?, ?, ?, ?, ?, ?);`,
		identity.Backend, identity.Version, identity.DocumentCount, identity.ChunkCount,
		identity.DocumentDigest, identity.ChunkDigest, identity.ContentDigest); err != nil {
		return errors.Wrap(err, "write content identity")
	}
	return nil
}

func writeDocuments(ctx context.Context, db *sql.DB, producer Producer[rag.Document]) (string, int, error) {
	count := 0
	digestValue, err := digest.JSONSequence(ctx, func(yield func(rag.Document) error) error {
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			return err
		}
		committed := false
		defer func() {
			if !committed {
				_ = tx.Rollback()
			}
		}()
		err = producer(ctx, func(document rag.Document) error {
			if err := rag.ValidateDocument(document); err != nil {
				return err
			}
			metadata, err := json.Marshal(document.Metadata)
			if err != nil {
				return errors.Wrap(err, "encode document metadata")
			}
			if _, err := tx.ExecContext(ctx, `INSERT INTO document VALUES (?, ?, ?, ?, ?, ?, ?)`, document.ID, count, document.SourceURI, document.Title, document.Text, document.ContentDigest, metadata); err != nil {
				return errors.Wrapf(err, "insert document %q", document.ID)
			}
			count++
			return yield(document)
		})
		if err != nil {
			return err
		}
		if err := tx.Commit(); err != nil {
			return err
		}
		committed = true
		return nil
	})
	return digestValue, count, err
}

func writeChunks(ctx context.Context, db *sql.DB, producer Producer[rag.Chunk]) (string, int, error) {
	count := 0
	digestValue, err := digest.JSONSequence(ctx, func(yield func(rag.Chunk) error) error {
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			return err
		}
		committed := false
		defer func() {
			if !committed {
				_ = tx.Rollback()
			}
		}()
		err = producer(ctx, func(chunk rag.Chunk) error {
			var sourceLength int
			var sourceSlice []byte
			if err := tx.QueryRowContext(ctx, `
SELECT length(CAST(text AS BLOB)),
       substr(CAST(text AS BLOB), ? + 1, ?)
FROM document WHERE id = ?`,
				chunk.Range.ByteStart, chunk.Range.ByteEnd-chunk.Range.ByteStart, chunk.DocumentID,
			).Scan(&sourceLength, &sourceSlice); err != nil {
				return errors.Wrapf(err, "load parent document source for chunk %q", chunk.ID)
			}
			if err := rag.ValidateChunkSource(chunk.DocumentID, sourceLength, sourceSlice, chunk); err != nil {
				return err
			}
			if _, err := tx.ExecContext(ctx, `INSERT INTO chunk VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`, chunk.ID, count, chunk.DocumentID, chunk.Ordinal, chunk.Range.ByteStart, chunk.Range.ByteEnd, chunk.Text, chunk.ContentDigest, chunk.Chunker); err != nil {
				return errors.Wrapf(err, "insert chunk %q", chunk.ID)
			}
			count++
			return yield(chunk)
		})
		if err != nil {
			return err
		}
		if err := tx.Commit(); err != nil {
			return err
		}
		committed = true
		return nil
	})
	return digestValue, count, err
}

func openDatabase(ctx context.Context, path string, readOnly bool) (*sql.DB, error) {
	parameters := url.Values{}
	if readOnly {
		parameters.Set("mode", "ro")
		parameters.Set("immutable", "1")
		parameters.Set("_query_only", "1")
		parameters.Set("_foreign_keys", "1")
	} else {
		parameters.Set("_foreign_keys", "1")
	}
	db, err := sql.Open("sqlite3", (&url.URL{Scheme: "file", Path: path, RawQuery: parameters.Encode()}).String())
	if err != nil {
		return nil, errors.Wrap(err, "open content SQLite database")
	}
	if err := db.PingContext(ctx); err != nil {
		_ = db.Close()
		return nil, errors.Wrap(err, "ping content SQLite database")
	}
	// A serving index is immutable and lookup batches are deliberately small.
	// Keep a small bounded pool so concurrent requests do not create an
	// unbounded number of SQLite connections and page caches.
	db.SetMaxOpenConns(4)
	db.SetMaxIdleConns(4)
	return db, nil
}

// Open opens and validates an immutable content database. Inspect streams
// canonical rows and therefore does not materialize the corpus.
func Open(ctx context.Context, config Config) (*Index, Identity, error) {
	if strings.TrimSpace(config.Path) == "" {
		return nil, Identity{}, errors.New("content database path is required")
	}
	maxBatch := config.MaxBatch
	if maxBatch == 0 {
		maxBatch = DefaultMaxBatch
	}
	if maxBatch < 1 {
		return nil, Identity{}, errors.New("content maximum batch must be positive")
	}
	db, err := openDatabase(ctx, config.Path, true)
	if err != nil {
		return nil, Identity{}, err
	}
	index := &Index{db: db, path: config.Path, maxBatch: maxBatch}
	identity, err := index.ReadIdentity(ctx)
	if err != nil {
		_ = db.Close()
		return nil, Identity{}, errors.Wrap(err, "inspect content database")
	}
	return index, identity, nil
}

// ReadIdentity reads the publisher's identity row without scanning payload
// text. Serving startup uses this bounded operation; Inspect is the explicit
// offline integrity sweep.
func (i *Index) ReadIdentity(ctx context.Context) (Identity, error) {
	if i == nil || i.db == nil {
		return Identity{}, errors.New("content index is not open")
	}
	var identity Identity
	if err := i.db.QueryRowContext(ctx, `
SELECT backend, version, document_count, chunk_count,
       document_digest, chunk_digest, content_digest
FROM content_identity WHERE id = 1`).Scan(
		&identity.Backend, &identity.Version, &identity.DocumentCount, &identity.ChunkCount,
		&identity.DocumentDigest, &identity.ChunkDigest, &identity.ContentDigest); err != nil {
		return Identity{}, errors.Wrap(err, "read content identity")
	}
	if identity.Backend != Backend || identity.Version != ManifestVersion ||
		identity.DocumentCount < 1 || identity.ChunkCount < 1 ||
		strings.TrimSpace(identity.DocumentDigest) == "" || strings.TrimSpace(identity.ChunkDigest) == "" ||
		strings.TrimSpace(identity.ContentDigest) == "" {
		return Identity{}, errors.New("content identity row is invalid")
	}
	return identity, nil
}

// Inspect verifies canonical counts and digests without retaining rows.
func (i *Index) Inspect(ctx context.Context) (Identity, error) {
	if i == nil || i.db == nil {
		return Identity{}, errors.New("content index is not open")
	}
	documentDigest, documentCount, err := streamDocuments(ctx, i.db, nil)
	if err != nil {
		return Identity{}, errors.Wrap(err, "digest content documents")
	}
	chunkDigest, chunkCount, err := streamChunks(ctx, i.db, nil)
	if err != nil {
		return Identity{}, errors.Wrap(err, "digest content chunks")
	}
	return NewIdentity(documentCount, chunkCount, documentDigest, chunkDigest)
}

func streamDocuments(ctx context.Context, db *sql.DB, yield func(rag.Document) error) (string, int, error) {
	count := 0
	digestValue, err := digest.JSONSequence(ctx, func(digestYield func(rag.Document) error) error {
		rows, err := db.QueryContext(ctx, `SELECT id, source_uri, title, text, content_digest, metadata_json FROM document ORDER BY ordinal`)
		if err != nil {
			return err
		}
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			document, err := scanDocument(rows)
			if err != nil {
				return err
			}
			if err := rag.ValidateDocument(document); err != nil {
				return err
			}
			if yield != nil {
				if err := yield(document); err != nil {
					return err
				}
			}
			if err := digestYield(document); err != nil {
				return err
			}
			count++
		}
		return rows.Err()
	})
	return digestValue, count, err
}

func streamChunks(ctx context.Context, db *sql.DB, yield func(rag.Chunk) error) (string, int, error) {
	count := 0
	digestValue, err := digest.JSONSequence(ctx, func(digestYield func(rag.Chunk) error) error {
		rows, err := db.QueryContext(ctx, `
SELECT c.id, c.document_id, c.document_ordinal, c.byte_start, c.byte_end,
       c.text, c.content_digest, c.chunker,
       length(CAST(d.text AS BLOB)),
       substr(CAST(d.text AS BLOB), c.byte_start + 1, c.byte_end - c.byte_start)
FROM chunk c
JOIN document d ON d.id = c.document_id
ORDER BY c.ordinal`)
		if err != nil {
			return err
		}
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			var chunk rag.Chunk
			var sourceLength int
			var sourceSlice []byte
			if err := rows.Scan(
				&chunk.ID, &chunk.DocumentID, &chunk.Ordinal, &chunk.Range.ByteStart,
				&chunk.Range.ByteEnd, &chunk.Text, &chunk.ContentDigest, &chunk.Chunker,
				&sourceLength, &sourceSlice,
			); err != nil {
				return err
			}
			if err := rag.ValidateChunkSource(chunk.DocumentID, sourceLength, sourceSlice, chunk); err != nil {
				return err
			}
			if yield != nil {
				if err := yield(chunk); err != nil {
					return err
				}
			}
			if err := digestYield(chunk); err != nil {
				return err
			}
			count++
		}
		return rows.Err()
	})
	return digestValue, count, err
}

// Documents returns exactly the requested documents in caller order.
func (i *Index) Documents(ctx context.Context, ids []string) ([]rag.Document, error) {
	if err := i.validateIDs(ids); err != nil {
		return nil, err
	}
	result := make([]rag.Document, 0, len(ids))
	for _, id := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		document, err := scanDocument(i.db.QueryRowContext(ctx,
			`SELECT id, source_uri, title, text, content_digest, metadata_json FROM document WHERE id = ?`, id))
		if err != nil {
			return nil, errors.Wrapf(err, "load content document %q", id)
		}
		result = append(result, document)
	}
	return result, nil
}

// Chunks returns exactly the requested chunks in caller order.
func (i *Index) Chunks(ctx context.Context, ids []string) ([]rag.Chunk, error) {
	if err := i.validateIDs(ids); err != nil {
		return nil, err
	}
	result := make([]rag.Chunk, 0, len(ids))
	for _, id := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		chunk, err := scanChunk(i.db.QueryRowContext(ctx,
			`SELECT id, document_id, document_ordinal, byte_start, byte_end, text, content_digest, chunker FROM chunk WHERE id = ?`, id))
		if err != nil {
			return nil, errors.Wrapf(err, "load content chunk %q", id)
		}
		result = append(result, chunk)
	}
	return result, nil
}

// CandidateMetadata returns the authorization-only projection in caller
// order. Source text is not selected by this query.
func (i *Index) CandidateMetadata(ctx context.Context, ids []string) ([]content.CandidateMetadata, error) {
	if err := i.validateIDs(ids); err != nil {
		return nil, err
	}
	result := make([]content.CandidateMetadata, 0, len(ids))
	for _, id := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var candidate content.CandidateMetadata
		var metadataJSON string
		if err := i.db.QueryRowContext(ctx, `
SELECT c.id, c.document_id, d.metadata_json
FROM chunk c JOIN document d ON d.id = c.document_id
WHERE c.id = ?`, id).Scan(&candidate.ChunkID, &candidate.DocumentID, &metadataJSON); err != nil {
			return nil, errors.Wrapf(err, "load content metadata %q", id)
		}
		if err := json.Unmarshal([]byte(metadataJSON), &candidate.Metadata); err != nil {
			return nil, errors.Wrapf(err, "decode content metadata %q", id)
		}
		result = append(result, candidate)
	}
	return result, nil
}

func (i *Index) validateIDs(ids []string) error {
	if i == nil || i.db == nil {
		return errors.New("content index is not open")
	}
	if len(ids) == 0 {
		return errors.New("content lookup requires at least one ID")
	}
	if len(ids) > i.maxBatch {
		return errors.Errorf("content lookup batch size %d exceeds maximum %d", len(ids), i.maxBatch)
	}
	seen := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		if strings.TrimSpace(id) == "" {
			return errors.New("content lookup ID cannot be empty")
		}
		if _, ok := seen[id]; ok {
			return errors.Errorf("content lookup contains duplicate ID %q", id)
		}
		seen[id] = struct{}{}
	}
	return nil
}

func (i *Index) Close() error {
	if i == nil || i.db == nil {
		return nil
	}
	err := i.db.Close()
	i.db = nil
	return err
}

type scanner interface {
	Scan(...any) error
}

func scanDocument(row scanner) (rag.Document, error) {
	var document rag.Document
	var metadataJSON string
	if err := row.Scan(&document.ID, &document.SourceURI, &document.Title, &document.Text, &document.ContentDigest, &metadataJSON); err != nil {
		return rag.Document{}, err
	}
	if err := json.Unmarshal([]byte(metadataJSON), &document.Metadata); err != nil {
		return rag.Document{}, errors.Wrap(err, "decode document metadata")
	}
	if err := rag.ValidateDocument(document); err != nil {
		return rag.Document{}, err
	}
	return document, nil
}

func scanChunk(row scanner) (rag.Chunk, error) {
	var chunk rag.Chunk
	if err := row.Scan(&chunk.ID, &chunk.DocumentID, &chunk.Ordinal, &chunk.Range.ByteStart, &chunk.Range.ByteEnd, &chunk.Text, &chunk.ContentDigest, &chunk.Chunker); err != nil {
		return rag.Chunk{}, err
	}
	return chunk, nil
}
