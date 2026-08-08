package gochunk

import (
	"fmt"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

const SymbolRepresentationVersion = "go-symbol-v1"

func Representations(records []Record) ([]rag.Representation, error) {
	chunks := make([]rag.Chunk, len(records))
	for index, record := range records {
		chunks[index] = record.Chunk
	}
	raw, err := rag.RawRepresentations(chunks)
	if err != nil {
		return nil, err
	}
	result := make([]rag.Representation, 0, len(records)*2)
	result = append(result, raw...)
	for _, record := range records {
		text := symbolText(record.Metadata)
		identity, err := digest.JSON(struct{ ChunkID, Kind, Version, TextDigest string }{record.Chunk.ID, "symbol", SymbolRepresentationVersion, digest.Text(text)})
		if err != nil {
			return nil, errors.Wrap(err, "build symbol representation identity")
		}
		result = append(result, rag.Representation{ID: "rep-" + identity[:16], ChunkID: record.Chunk.ID, Kind: "symbol", Text: text, ContentDigest: digest.Text(text)})
	}
	return result, nil
}

func symbolText(metadata Metadata) string {
	kind := metadata.Kind
	if kind == "" {
		kind = "declaration"
	}
	name := metadata.QualifiedName
	if name == "" {
		name = metadata.Name
	}
	lines := []string{fmt.Sprintf("Go %s %s.", kind, name)}
	if metadata.PackagePath != "" {
		lines = append(lines, "Package: "+metadata.PackagePath+".")
	}
	if metadata.Signature != "" {
		lines = append(lines, "Signature: "+metadata.Signature+".")
	}
	lines = append(lines, "Repository: "+metadata.Repository+".", "Defined in "+metadata.Path+".")
	if metadata.Documentation != "" {
		lines = append(lines, "Documentation: "+metadata.Documentation)
	}
	return strings.Join(lines, "\n")
}
