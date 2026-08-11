package indexbundle

import (
	"bufio"
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/url"
	"os"
	"sort"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	bleveindex "github.com/go-go-golems/ragkit/rag/lexical/bleve"
	"github.com/go-go-golems/ragkit/rag/vector/sqliteexact"
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
	Chunker   ChunkerIdentity
	Embedding *VectorIdentity
}

type buildPlan struct {
	BundleID             string
	CorpusDigest         string
	ChunkDigest          string
	RepresentationDigest string
	RepresentationKinds  []string
	DocumentCount        int
	ChunkCount           int
	RepresentationCount  int
	Lexical              BackendIdentity
	Vector               *VectorIdentity
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
	 title TEXT NOT NULL,
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
		if _, err := tx.ExecContext(ctx, `INSERT INTO document VALUES (?, ?, ?, ?)`, document.ID, start+index, document.Title, encoded); err != nil {
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
		if chunk.Chunker != k.spec.Chunker.Name {
			_ = tx.Rollback()
			return errors.Errorf("chunk %q uses chunker %q but staging declares %q", chunk.ID, chunk.Chunker, k.spec.Chunker.Name)
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

func (k *stagingKernel) seal(ctx context.Context) (buildPlan, error) {
	if err := ctx.Err(); err != nil {
		return buildPlan{}, err
	}
	wantPhase := stagingRepresentations
	if k.spec.Embedding != nil {
		wantPhase = stagingVectors
	}
	if k.phase != wantPhase || k.documents == 0 || k.chunks == 0 || k.reps == 0 {
		return buildPlan{}, errors.Errorf("cannot seal incomplete staging relation in phase %d", k.phase)
	}
	if k.spec.Embedding != nil && k.vectors != k.reps {
		return buildPlan{}, errors.Errorf("staged vector count %d differs from representation count %d", k.vectors, k.reps)
	}

	corpusDigest, err := stagedCanonicalDigest[rag.Document](ctx, k.db, "document")
	if err != nil {
		return buildPlan{}, errors.Wrap(err, "digest staged corpus")
	}
	chunkDigest, err := stagedCanonicalDigest[rag.Chunk](ctx, k.db, "chunk")
	if err != nil {
		return buildPlan{}, errors.Wrap(err, "digest staged chunks")
	}
	representationDigest, err := stagedCanonicalDigest[rag.Representation](ctx, k.db, "representation")
	if err != nil {
		return buildPlan{}, errors.Wrap(err, "digest staged representations")
	}
	lexicalDigest, err := k.lexicalDigest(ctx)
	if err != nil {
		return buildPlan{}, errors.Wrap(err, "digest staged lexical records")
	}
	lexical := BackendIdentity{
		Backend: "bleve-bm25", Version: bleveindex.ManifestVersion, Channel: "bm25",
		TitleBoost: 2, BodyBoost: 1, ContentDigest: lexicalDigest,
	}
	kinds, err := k.representationKinds(ctx)
	if err != nil {
		return buildPlan{}, err
	}
	vector := cloneVectorIdentity(k.spec.Embedding)
	if vector != nil {
		vector.RepresentationDigest = representationDigest
		vector.ContentDigest, err = k.vectorDigest(ctx)
		if err != nil {
			return buildPlan{}, errors.Wrap(err, "digest staged vector records")
		}
	}
	bundleID, err := calculateIDFromDigests(
		corpusDigest, chunkDigest, representationDigest, kinds, k.spec.Chunker, lexical, vector,
	)
	if err != nil {
		return buildPlan{}, err
	}
	k.phase = stagingSealed
	return buildPlan{
		BundleID: bundleID, CorpusDigest: corpusDigest, ChunkDigest: chunkDigest,
		RepresentationDigest: representationDigest, RepresentationKinds: kinds,
		DocumentCount: k.documents, ChunkCount: k.chunks, RepresentationCount: k.reps,
		Lexical: lexical, Vector: vector,
	}, nil
}

func stagedCanonicalDigest[T any](ctx context.Context, db *sql.DB, table string) (string, error) {
	if table != "document" && table != "chunk" && table != "representation" {
		return "", errors.Errorf("unsupported staged digest table %q", table)
	}
	return digest.JSONSequence(ctx, func(yield func(T) error) error {
		rows, err := db.QueryContext(ctx, `SELECT canonical_json FROM `+table+` ORDER BY ordinal`)
		if err != nil {
			return err
		}
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			var encoded []byte
			if err := rows.Scan(&encoded); err != nil {
				return err
			}
			var value T
			if err := json.Unmarshal(encoded, &value); err != nil {
				return err
			}
			if err := yield(value); err != nil {
				return err
			}
		}
		return rows.Err()
	})
}

type stagedLexicalRecord struct {
	RepresentationID string `json:"representation_id"`
	ChunkID          string `json:"chunk_id"`
	DocumentID       string `json:"document_id"`
	Kind             string `json:"kind"`
	Title            string `json:"title"`
	Body             string `json:"body"`
}

func (k *stagingKernel) lexicalDigest(ctx context.Context) (string, error) {
	return digest.JSONSequence(ctx, func(yield func(stagedLexicalRecord) error) error {
		rows, err := k.db.QueryContext(ctx, `
SELECT r.id, r.chunk_id, c.document_id, r.kind,
       d.title, r.text
FROM representation r
JOIN chunk c ON c.id = r.chunk_id
JOIN document d ON d.id = c.document_id
ORDER BY r.id`)
		if err != nil {
			return err
		}
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			var record stagedLexicalRecord
			if err := rows.Scan(&record.RepresentationID, &record.ChunkID, &record.DocumentID, &record.Kind, &record.Title, &record.Body); err != nil {
				return err
			}
			if err := yield(record); err != nil {
				return err
			}
		}
		return rows.Err()
	})
}

func (k *stagingKernel) produceLexical(ctx context.Context, yield func(bleveindex.Record) error) error {
	if k.phase != stagingSealed {
		return errors.New("staging relation must be sealed before reading lexical records")
	}
	rows, err := k.db.QueryContext(ctx, `
SELECT r.id, r.chunk_id, c.document_id, r.kind, d.title, r.text
FROM representation r
JOIN chunk c ON c.id = r.chunk_id
JOIN document d ON d.id = c.document_id
ORDER BY r.id`)
	if err != nil {
		return errors.Wrap(err, "read staged lexical records")
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var record bleveindex.Record
		if err := rows.Scan(&record.RepresentationID, &record.ChunkID, &record.DocumentID, &record.Kind, &record.Title, &record.Body); err != nil {
			return errors.Wrap(err, "scan staged lexical record")
		}
		if err := yield(record); err != nil {
			return err
		}
	}
	return rows.Err()
}

func (k *stagingKernel) vectorDigest(ctx context.Context) (string, error) {
	return digest.JSONSequence(ctx, func(yield func(sqliteexact.Entry) error) error {
		rows, err := k.db.QueryContext(ctx, `
SELECT r.id, r.chunk_id, c.document_id, v.dimensions, v.values_blob, r.content_digest
FROM vector v
JOIN representation r ON r.id = v.representation_id
JOIN chunk c ON c.id = r.chunk_id
ORDER BY r.id`)
		if err != nil {
			return err
		}
		defer func() { _ = rows.Close() }()
		for rows.Next() {
			var entry sqliteexact.Entry
			var dimensions int
			var blob []byte
			if err := rows.Scan(&entry.RepresentationID, &entry.ChunkID, &entry.DocumentID, &dimensions, &blob, &entry.ContentDigest); err != nil {
				return err
			}
			values, err := decodeStagedVector(blob, dimensions)
			if err != nil {
				return err
			}
			entry.Values = values
			if err := yield(entry); err != nil {
				return err
			}
		}
		return rows.Err()
	})
}

func (k *stagingKernel) produceVectors(ctx context.Context, yield func(sqliteexact.Entry) error) error {
	if k.phase != stagingSealed || k.spec.Embedding == nil {
		return errors.New("sealed vector staging relation is required")
	}
	rows, err := k.db.QueryContext(ctx, `
SELECT r.id, r.chunk_id, c.document_id, v.dimensions, v.values_blob, r.content_digest
FROM vector v
JOIN representation r ON r.id = v.representation_id
JOIN chunk c ON c.id = r.chunk_id
ORDER BY r.id`)
	if err != nil {
		return errors.Wrap(err, "read staged vector records")
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var entry sqliteexact.Entry
		var dimensions int
		var blob []byte
		if err := rows.Scan(&entry.RepresentationID, &entry.ChunkID, &entry.DocumentID, &dimensions, &blob, &entry.ContentDigest); err != nil {
			return errors.Wrap(err, "scan staged vector record")
		}
		values, err := decodeStagedVector(blob, dimensions)
		if err != nil {
			return errors.Wrapf(err, "decode staged vector record %q", entry.RepresentationID)
		}
		entry.Values = values
		if err := yield(entry); err != nil {
			return err
		}
	}
	return rows.Err()
}

func (k *stagingKernel) writeJSONArray(ctx context.Context, path, table string) error {
	if k.phase != stagingSealed {
		return errors.New("staging relation must be sealed before writing payloads")
	}
	if table != "chunk" && table != "representation" {
		return errors.Errorf("unsupported staged payload table %q", table)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return errors.Wrap(err, "create staged JSON payload")
	}
	closeFile := true
	defer func() {
		if closeFile {
			_ = file.Close()
		}
	}()
	writer := bufio.NewWriterSize(file, 64*1024)
	if err := writer.WriteByte('['); err != nil {
		return errors.Wrap(err, "start staged JSON payload")
	}
	rows, err := k.db.QueryContext(ctx, `SELECT canonical_json FROM `+table+` ORDER BY ordinal`)
	if err != nil {
		return errors.Wrap(err, "read staged JSON payload")
	}
	defer func() { _ = rows.Close() }()
	count := 0
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return err
		}
		var encoded []byte
		if err := rows.Scan(&encoded); err != nil {
			return errors.Wrap(err, "scan staged JSON payload")
		}
		if count > 0 {
			if err := writer.WriteByte(','); err != nil {
				return errors.Wrap(err, "separate staged JSON payload")
			}
		}
		if _, err := writer.Write(encoded); err != nil {
			return errors.Wrap(err, "write staged JSON payload value")
		}
		count++
	}
	if err := rows.Err(); err != nil {
		return errors.Wrap(err, "iterate staged JSON payload")
	}
	if err := writer.WriteByte(']'); err != nil {
		return errors.Wrap(err, "finish staged JSON payload")
	}
	if err := writer.WriteByte('\n'); err != nil {
		return errors.Wrap(err, "terminate staged JSON payload")
	}
	if err := writer.Flush(); err != nil {
		return errors.Wrap(err, "flush staged JSON payload")
	}
	if err := file.Sync(); err != nil {
		return errors.Wrap(err, "sync staged JSON payload")
	}
	if err := file.Close(); err != nil {
		return errors.Wrap(err, "close staged JSON payload")
	}
	closeFile = false
	return nil
}

func (k *stagingKernel) representationKinds(ctx context.Context) ([]string, error) {
	rows, err := k.db.QueryContext(ctx, `SELECT DISTINCT kind FROM representation ORDER BY kind`)
	if err != nil {
		return nil, errors.Wrap(err, "read staged representation kinds")
	}
	defer func() { _ = rows.Close() }()
	var kinds []string
	for rows.Next() {
		var kind string
		if err := rows.Scan(&kind); err != nil {
			return nil, errors.Wrap(err, "scan staged representation kind")
		}
		kinds = append(kinds, kind)
	}
	if err := rows.Err(); err != nil {
		return nil, errors.Wrap(err, "iterate staged representation kinds")
	}
	sort.Strings(kinds)
	return kinds, nil
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

func decodeStagedVector(encoded []byte, dimensions int) ([]float32, error) {
	if dimensions < 1 || len(encoded) != dimensions*4 {
		return nil, errors.Errorf("invalid staged vector blob length %d for %d dimensions", len(encoded), dimensions)
	}
	values := make([]float32, dimensions)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(encoded[index*4:]))
	}
	return values, nil
}
