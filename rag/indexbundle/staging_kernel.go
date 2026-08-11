package indexbundle

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/url"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
	vectorutil "github.com/go-go-golems/ragkit/vector"
	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"
)

type stagingPhase uint8

const (
	stagingDocuments stagingPhase = iota
	stagingChunks
	stagingRepresentations
	stagingVectors
	stagingSealed
)

type stagingSpec struct {
	BatchSize int
	Embedding *VectorIdentity
}

// Stager is the bounded, fail-closed input boundary used by BuildStream.
// Calls copy their batch into durable staging before returning.
type Stager struct {
	kernel *stagingKernel
}

func (s *Stager) AddDocuments(ctx context.Context, batch []rag.Document) error {
	return s.kernel.addDocuments(ctx, batch)
}

func (s *Stager) AddChunks(ctx context.Context, batch []rag.Chunk) error {
	return s.kernel.addChunks(ctx, batch)
}

func (s *Stager) AddRepresentations(ctx context.Context, batch []rag.Representation) error {
	return s.kernel.addRepresentations(ctx, batch)
}

func (s *Stager) AddVectors(ctx context.Context, batch []rag.Vector) error {
	return s.kernel.addVectors(ctx, batch)
}

type stagingKernel struct {
	db        *sql.DB
	spec      stagingSpec
	phase     stagingPhase
	documents int
	chunks    int
	reps      int
	vectors   int
}

func openStagingKernel(ctx context.Context, path string, spec stagingSpec) (*stagingKernel, error) {
	if spec.BatchSize < 1 {
		return nil, errors.New("staging batch size must be positive")
	}
	dsn := (&url.URL{Scheme: "file", Path: path, RawQuery: url.Values{
		"_foreign_keys": {"on"},
	}.Encode()}).String()
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, errors.Wrap(err, "open bundle staging database")
	}
	if _, err := db.ExecContext(ctx, `
CREATE TABLE document (
 id TEXT PRIMARY KEY,
 ordinal INTEGER NOT NULL UNIQUE,
 canonical_json BLOB NOT NULL
);
CREATE TABLE chunk (
 id TEXT PRIMARY KEY,
 ordinal INTEGER NOT NULL UNIQUE,
 document_id TEXT NOT NULL REFERENCES document(id),
 document_ordinal INTEGER NOT NULL,
 canonical_json BLOB NOT NULL,
 UNIQUE(document_id, document_ordinal)
);
CREATE TABLE representation (
 id TEXT PRIMARY KEY,
 ordinal INTEGER NOT NULL UNIQUE,
 chunk_id TEXT NOT NULL REFERENCES chunk(id),
 kind TEXT NOT NULL,
 text TEXT NOT NULL,
 content_digest TEXT NOT NULL,
 canonical_json BLOB NOT NULL
);
CREATE TABLE vector (
 representation_id TEXT PRIMARY KEY REFERENCES representation(id),
 model TEXT NOT NULL,
 dimensions INTEGER NOT NULL,
 values_blob BLOB NOT NULL
);`); err != nil {
		_ = db.Close()
		return nil, errors.Wrap(err, "create bundle staging schema")
	}
	return &stagingKernel{db: db, spec: spec, phase: stagingDocuments}, nil
}

func (k *stagingKernel) close() error {
	if k == nil || k.db == nil {
		return nil
	}
	return k.db.Close()
}

func (k *stagingKernel) addDocuments(ctx context.Context, batch []rag.Document) error {
	if err := k.requireBatch(ctx, stagingDocuments, len(batch)); err != nil {
		return err
	}
	tx, err := k.db.BeginTx(ctx, nil)
	if err != nil {
		return errors.Wrap(err, "begin staged document batch")
	}
	start := k.documents
	for index, document := range batch {
		if err := rag.ValidateDocument(document); err != nil {
			_ = tx.Rollback()
			return errors.Wrap(err, "validate staged document")
		}
		encoded, err := json.Marshal(document)
		if err != nil {
			_ = tx.Rollback()
			return errors.Wrap(err, "encode staged document")
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO document VALUES (?, ?, ?)`, document.ID, start+index, encoded); err != nil {
			_ = tx.Rollback()
			return errors.Wrapf(err, "stage document %q", document.ID)
		}
	}
	if err := tx.Commit(); err != nil {
		return errors.Wrap(err, "commit staged document batch")
	}
	k.documents += len(batch)
	return nil
}

func (k *stagingKernel) addChunks(ctx context.Context, batch []rag.Chunk) error {
	if err := k.requireBatch(ctx, stagingChunks, len(batch)); err != nil {
		return err
	}
	tx, err := k.db.BeginTx(ctx, nil)
	if err != nil {
		return errors.Wrap(err, "begin staged chunk batch")
	}
	start := k.chunks
	for index, chunk := range batch {
		document, err := loadStagedJSON[rag.Document](ctx, tx, `SELECT canonical_json FROM document WHERE id = ?`, chunk.DocumentID)
		if err != nil {
			_ = tx.Rollback()
			return errors.Wrapf(err, "load parent document for chunk %q", chunk.ID)
		}
		if chunk.Ordinal < 0 {
			_ = tx.Rollback()
			return errors.Errorf("chunk %q has negative ordinal %d", chunk.ID, chunk.Ordinal)
		}
		if err := rag.ValidateChunk(document, chunk); err != nil {
			_ = tx.Rollback()
			return errors.Wrap(err, "validate staged chunk")
		}
		encoded, err := json.Marshal(chunk)
		if err != nil {
			_ = tx.Rollback()
			return errors.Wrap(err, "encode staged chunk")
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO chunk VALUES (?, ?, ?, ?, ?)`, chunk.ID, start+index, chunk.DocumentID, chunk.Ordinal, encoded); err != nil {
			_ = tx.Rollback()
			return errors.Wrapf(err, "stage chunk %q", chunk.ID)
		}
	}
	if err := tx.Commit(); err != nil {
		return errors.Wrap(err, "commit staged chunk batch")
	}
	k.chunks += len(batch)
	k.phase = stagingChunks
	return nil
}

func (k *stagingKernel) addRepresentations(ctx context.Context, batch []rag.Representation) error {
	if err := k.requireBatch(ctx, stagingRepresentations, len(batch)); err != nil {
		return err
	}
	tx, err := k.db.BeginTx(ctx, nil)
	if err != nil {
		return errors.Wrap(err, "begin staged representation batch")
	}
	start := k.reps
	for index, representation := range batch {
		chunk, err := loadStagedJSON[rag.Chunk](ctx, tx, `SELECT canonical_json FROM chunk WHERE id = ?`, representation.ChunkID)
		if err != nil {
			_ = tx.Rollback()
			return errors.Wrapf(err, "load parent chunk for representation %q", representation.ID)
		}
		if err := rag.ValidateRepresentations([]rag.Chunk{chunk}, []rag.Representation{representation}); err != nil {
			_ = tx.Rollback()
			return errors.Wrap(err, "validate staged representation")
		}
		encoded, err := json.Marshal(representation)
		if err != nil {
			_ = tx.Rollback()
			return errors.Wrap(err, "encode staged representation")
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO representation VALUES (?, ?, ?, ?, ?, ?, ?)`,
			representation.ID, start+index, representation.ChunkID, representation.Kind,
			representation.Text, representation.ContentDigest, encoded); err != nil {
			_ = tx.Rollback()
			return errors.Wrapf(err, "stage representation %q", representation.ID)
		}
	}
	if err := tx.Commit(); err != nil {
		return errors.Wrap(err, "commit staged representation batch")
	}
	k.reps += len(batch)
	k.phase = stagingRepresentations
	return nil
}

func (k *stagingKernel) addVectors(ctx context.Context, batch []rag.Vector) error {
	if k.spec.Embedding == nil {
		return errors.New("cannot stage vectors for a lexical-only build")
	}
	if err := k.requireBatch(ctx, stagingVectors, len(batch)); err != nil {
		return err
	}
	tx, err := k.db.BeginTx(ctx, nil)
	if err != nil {
		return errors.Wrap(err, "begin staged vector batch")
	}
	for _, vector := range batch {
		if strings.TrimSpace(vector.RepresentationID) == "" {
			_ = tx.Rollback()
			return errors.New("staged vector representation ID is required")
		}
		if vector.Model != k.spec.Embedding.Model || len(vector.Values) != k.spec.Embedding.Dimensions {
			_ = tx.Rollback()
			return errors.Errorf("vector %q differs from staged embedding identity", vector.RepresentationID)
		}
		if err := vectorutil.ValidateFinite(vector.Values); err != nil {
			_ = tx.Rollback()
			return errors.Wrapf(err, "validate staged vector %q", vector.RepresentationID)
		}
		blob := encodeStagedVector(vector.Values)
		if _, err := tx.ExecContext(ctx, `INSERT INTO vector VALUES (?, ?, ?, ?)`,
			vector.RepresentationID, vector.Model, len(vector.Values), blob); err != nil {
			_ = tx.Rollback()
			return errors.Wrapf(err, "stage vector %q", vector.RepresentationID)
		}
	}
	if err := tx.Commit(); err != nil {
		return errors.Wrap(err, "commit staged vector batch")
	}
	k.vectors += len(batch)
	k.phase = stagingVectors
	return nil
}

func (k *stagingKernel) requireBatch(ctx context.Context, requested stagingPhase, length int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if k.phase == stagingSealed || requested < k.phase || requested > k.phase+1 {
		return errors.Errorf("invalid staging transition from %d to %d", k.phase, requested)
	}
	if requested > k.phase && !k.currentPhaseHasRows() {
		return errors.Errorf("invalid staging transition from empty phase %d to %d", k.phase, requested)
	}
	if length < 1 || length > k.spec.BatchSize {
		return errors.Errorf("staging batch size %d is outside [1,%d]", length, k.spec.BatchSize)
	}
	return nil
}

func (k *stagingKernel) currentPhaseHasRows() bool {
	switch k.phase {
	case stagingDocuments:
		return k.documents > 0
	case stagingChunks:
		return k.chunks > 0
	case stagingRepresentations:
		return k.reps > 0
	case stagingVectors:
		return k.vectors > 0
	default:
		return false
	}
}

type queryRower interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func loadStagedJSON[T any](ctx context.Context, db queryRower, query string, arguments ...any) (T, error) {
	var zero T
	var encoded []byte
	if err := db.QueryRowContext(ctx, query, arguments...).Scan(&encoded); err != nil {
		return zero, err
	}
	if err := json.Unmarshal(encoded, &zero); err != nil {
		return zero, err
	}
	return zero, nil
}

func encodeStagedVector(values []float32) []byte {
	encoded := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(encoded[index*4:], math.Float32bits(value))
	}
	return encoded
}
