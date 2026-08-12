package indexbundle

import (
	"context"
	"database/sql"
	"net/url"
	"os"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"
)

const verificationBatchSize = 512

// verificationRelation is verifier-owned scratch storage for exact identity
// and lineage checks. Its heap use is bounded by SQLite's configured page cache
// rather than by bundle cardinality.
type verificationRelation struct {
	db      *sql.DB
	path    string
	tx      *sql.Tx
	phase   verificationPhase
	pending int
}

type verificationPhase uint8

const (
	verificationChunks verificationPhase = iota
	verificationRepresentations
)

func openVerificationRelation(ctx context.Context) (*verificationRelation, error) {
	file, err := os.CreateTemp("", ".ragkit-verify-*.sqlite")
	if err != nil {
		return nil, errors.Wrap(err, "create bundle verification relation")
	}
	path := file.Name()
	if err := file.Close(); err != nil {
		_ = os.Remove(path)
		return nil, errors.Wrap(err, "close bundle verification relation file")
	}
	parameters := url.Values{
		"_busy_timeout": {"5000"},
		"_foreign_keys": {"on"},
	}
	db, err := sql.Open("sqlite3", (&url.URL{Scheme: "file", Path: path, RawQuery: parameters.Encode()}).String())
	if err != nil {
		_ = os.Remove(path)
		return nil, errors.Wrap(err, "open bundle verification relation")
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	closeAndRemove := func() {
		_ = db.Close()
		_ = os.Remove(path)
	}
	if _, err := db.ExecContext(ctx, `
PRAGMA journal_mode = OFF;
PRAGMA synchronous = OFF;
PRAGMA cache_size = -2048;
PRAGMA temp_store = FILE;
CREATE TABLE chunk_identity (
 id TEXT PRIMARY KEY,
 document_id TEXT NOT NULL,
 document_ordinal INTEGER NOT NULL,
 content_digest TEXT NOT NULL,
 UNIQUE(document_id, document_ordinal)
) WITHOUT ROWID;
CREATE INDEX chunk_identity_document ON chunk_identity(document_id);
CREATE TABLE representation_identity (
 id TEXT PRIMARY KEY
) WITHOUT ROWID;`); err != nil {
		closeAndRemove()
		return nil, errors.Wrap(err, "initialize bundle verification relation")
	}
	relation := &verificationRelation{db: db, path: path, phase: verificationChunks}
	if err := relation.begin(ctx); err != nil {
		closeAndRemove()
		return nil, err
	}
	return relation, nil
}

func (r *verificationRelation) begin(ctx context.Context) error {
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return errors.Wrap(err, "begin bundle verification batch")
	}
	r.tx = tx
	r.pending = 0
	return nil
}

func (r *verificationRelation) flush(ctx context.Context) error {
	if r.tx == nil {
		return nil
	}
	if err := r.tx.Commit(); err != nil {
		return errors.Wrap(err, "commit bundle verification batch")
	}
	r.tx = nil
	r.pending = 0
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}

func (r *verificationRelation) rotate(ctx context.Context) error {
	if r.pending < verificationBatchSize {
		return nil
	}
	if err := r.flush(ctx); err != nil {
		return err
	}
	return r.begin(ctx)
}

func (r *verificationRelation) addChunk(ctx context.Context, id, documentID string, ordinal int, contentDigest string) error {
	result, err := r.tx.ExecContext(ctx, `
INSERT OR IGNORE INTO chunk_identity(id, document_id, document_ordinal, content_digest)
VALUES (?, ?, ?, ?)`, id, documentID, ordinal, contentDigest)
	if err != nil {
		return errors.Wrapf(err, "record bundle chunk %q", id)
	}
	inserted, err := result.RowsAffected()
	if err != nil {
		return errors.Wrap(err, "inspect bundle chunk insertion")
	}
	if inserted == 0 {
		var exists int
		if err := r.tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM chunk_identity WHERE id = ?)`, id).Scan(&exists); err != nil {
			return errors.Wrap(err, "classify duplicate bundle chunk")
		}
		if exists != 0 {
			return errors.Errorf("bundle contains duplicate chunk %q", id)
		}
		return errors.Errorf("bundle contains duplicate ordinal %d for document %q", ordinal, documentID)
	}
	r.pending++
	return r.rotate(ctx)
}

func (r *verificationRelation) finishChunks(ctx context.Context) error {
	if r.phase != verificationChunks {
		return errors.New("bundle verification chunk relation is already complete")
	}
	if err := r.flush(ctx); err != nil {
		return err
	}
	r.phase = verificationRepresentations
	return r.begin(ctx)
}

func (r *verificationRelation) documentCount(ctx context.Context) (int, error) {
	var count int
	if err := r.tx.QueryRowContext(ctx, `SELECT COUNT(DISTINCT document_id) FROM chunk_identity`).Scan(&count); err != nil {
		return 0, errors.Wrap(err, "count bundle chunk documents")
	}
	return count, nil
}

func (r *verificationRelation) addRepresentation(ctx context.Context, id string) error {
	result, err := r.tx.ExecContext(ctx, `INSERT OR IGNORE INTO representation_identity(id) VALUES (?)`, id)
	if err != nil {
		return errors.Wrapf(err, "record bundle representation %q", id)
	}
	inserted, err := result.RowsAffected()
	if err != nil {
		return errors.Wrap(err, "inspect bundle representation insertion")
	}
	if inserted == 0 {
		return errors.Errorf("duplicate representation ID %q", id)
	}
	r.pending++
	return r.rotate(ctx)
}

func (r *verificationRelation) chunkContentDigest(ctx context.Context, id string) (string, bool, error) {
	var contentDigest string
	err := r.tx.QueryRowContext(ctx, `SELECT content_digest FROM chunk_identity WHERE id = ?`, id).Scan(&contentDigest)
	if err == sql.ErrNoRows {
		return "", false, nil
	}
	if err != nil {
		return "", false, errors.Wrapf(err, "look up bundle chunk %q", id)
	}
	return contentDigest, true, nil
}

func (r *verificationRelation) finishRepresentations(ctx context.Context) error {
	if r.phase != verificationRepresentations {
		return errors.New("bundle verification representation relation is unavailable")
	}
	return r.flush(ctx)
}

func (r *verificationRelation) closeAndRemove() error {
	if r == nil {
		return nil
	}
	if r.tx != nil {
		_ = r.tx.Rollback()
		r.tx = nil
	}
	var closeErr error
	if r.db != nil {
		closeErr = r.db.Close()
		r.db = nil
	}
	removeErr := os.Remove(r.path)
	if closeErr != nil {
		return errors.Wrap(closeErr, "close bundle verification relation")
	}
	if removeErr != nil && !os.IsNotExist(removeErr) {
		return errors.Wrap(removeErr, "remove bundle verification relation")
	}
	return nil
}
