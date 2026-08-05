// Package indexbundle builds and opens immutable, identity-checked persistent
// retrieval indexes.
package indexbundle

import (
	"sync"
	"time"

	"github.com/go-go-golems/ragkit/rag"
	bleveindex "github.com/go-go-golems/ragkit/rag/lexical/bleve"
)

const SchemaVersion = 1

type ChunkerIdentity struct {
	Name            string `json:"name"`
	MaximumRunes    int    `json:"maximum_runes"`
	OverlapRunes    int    `json:"overlap_runes"`
	MinSectionRunes int    `json:"min_section_runes,omitempty"`
}

type BackendIdentity struct {
	Backend    string  `json:"backend"`
	Version    int     `json:"version"`
	Channel    string  `json:"channel"`
	TitleBoost float64 `json:"title_boost,omitempty"`
	BodyBoost  float64 `json:"body_boost,omitempty"`
}

type VectorIdentity struct {
	Backend              string `json:"backend"`
	Version              int    `json:"version"`
	Channel              string `json:"channel"`
	Provider             string `json:"provider"`
	Model                string `json:"model"`
	Dimensions           int    `json:"dimensions"`
	RepresentationDigest string `json:"representation_digest"`
}

type Manifest struct {
	SchemaVersion       int             `json:"schema_version"`
	BundleID            string          `json:"bundle_id"`
	CreatedAt           time.Time       `json:"created_at"`
	CorpusDigest        string          `json:"corpus_digest"`
	CorpusPath          string          `json:"corpus_path"`
	DocumentCount       int             `json:"document_count"`
	ChunkCount          int             `json:"chunk_count"`
	RepresentationCount int             `json:"representation_count"`
	Chunker             ChunkerIdentity `json:"chunker"`
	RepresentationKinds []string        `json:"representation_kinds"`
	Lexical             BackendIdentity `json:"lexical"`
	Vector              *VectorIdentity `json:"vector,omitempty"`
}

type BuildInput struct {
	OutputRoot      string
	CorpusPath      string
	Documents       []rag.Document
	Chunks          []rag.Chunk
	Representations []rag.Representation
	Vectors         []rag.Vector
	QueryEmbedder   rag.Embedder
	Chunker         ChunkerIdentity
	Embedding       *VectorIdentity
}

type BuildResult struct {
	Manifest    Manifest `json:"manifest"`
	Path        string   `json:"path"`
	Reused      bool     `json:"reused"`
	BleveBytes  int64    `json:"bleve_bytes"`
	VectorBytes int64    `json:"vector_bytes"`
}

type OpenOptions struct {
	Path                string
	QueryEmbedder       rag.Embedder
	EmbeddingModel      string
	EmbeddingDimensions int
}

type Bundle struct {
	Manifest        Manifest
	Chunks          []rag.Chunk
	Representations []rag.Representation
	Lexical         rag.Index
	Vector          rag.Index

	closeOnce sync.Once
	closeErr  error
}

func (b *Bundle) Close() error {
	if b == nil {
		return nil
	}
	b.closeOnce.Do(func() {
		if b.Lexical != nil {
			b.closeErr = b.Lexical.Close()
		}
		if b.Vector != nil {
			if err := b.Vector.Close(); b.closeErr == nil {
				b.closeErr = err
			}
		}
	})
	return b.closeErr
}

func lexicalIdentity(manifest bleveindex.Manifest) BackendIdentity {
	return BackendIdentity{
		Backend: manifest.Backend, Version: manifest.Version,
		Channel: manifest.Channel, TitleBoost: manifest.TitleBoost,
		BodyBoost: manifest.BodyBoost,
	}
}
