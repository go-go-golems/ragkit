package gochunk

import (
	"fmt"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
)

const ExtractorVersion = "go-ast-v1"

type Snapshot struct {
	WorkspacePath string           `json:"workspace_path"`
	Repositories  []RepositoryHead `json:"repositories"`
	Digest        string           `json:"digest"`
}

type RepositoryHead struct {
	Name   string `json:"name"`
	Path   string `json:"path"`
	Commit string `json:"commit"`
	Tree   string `json:"tree"`
}

type Metadata struct {
	Repository    string `json:"repository"`
	Commit        string `json:"commit"`
	Path          string `json:"path"`
	PackageName   string `json:"package_name"`
	PackagePath   string `json:"package_path,omitempty"`
	Kind          string `json:"kind"`
	Name          string `json:"name,omitempty"`
	QualifiedName string `json:"qualified_name,omitempty"`
	Receiver      string `json:"receiver,omitempty"`
	Signature     string `json:"signature,omitempty"`
	Documentation string `json:"documentation,omitempty"`
	Exported      bool   `json:"exported,omitempty"`
	Test          bool   `json:"test,omitempty"`
	ByteStart     int    `json:"byte_start"`
	ByteEnd       int    `json:"byte_end"`
	LineStart     int    `json:"line_start"`
	LineEnd       int    `json:"line_end"`
	ParseStatus   string `json:"parse_status"`
}

type Record struct {
	Chunk    rag.Chunk `json:"chunk"`
	Metadata Metadata  `json:"metadata"`
}

type Diagnostic struct {
	DocumentID string `json:"document_id"`
	Path       string `json:"path"`
	Message    string `json:"message"`
}

func (record Record) Validate(document rag.Document) error {
	if err := rag.ValidateChunk(document, record.Chunk); err != nil {
		return err
	}
	metadata := record.Metadata
	if strings.TrimSpace(metadata.Repository) == "" || strings.TrimSpace(metadata.Commit) == "" || strings.TrimSpace(metadata.Path) == "" {
		return fmt.Errorf("chunk %q requires repository, commit, and path metadata", record.Chunk.ID)
	}
	if metadata.ByteStart != record.Chunk.Range.ByteStart || metadata.ByteEnd != record.Chunk.Range.ByteEnd {
		return fmt.Errorf("chunk %q metadata range differs from source range", record.Chunk.ID)
	}
	if metadata.LineStart < 1 || metadata.LineEnd < metadata.LineStart {
		return fmt.Errorf("chunk %q has invalid line range", record.Chunk.ID)
	}
	if strings.TrimSpace(metadata.Kind) == "" || strings.TrimSpace(metadata.PackageName) == "" {
		return fmt.Errorf("chunk %q requires kind and package metadata", record.Chunk.ID)
	}
	return nil
}
