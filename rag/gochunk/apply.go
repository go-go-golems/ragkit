package gochunk

import (
	"context"
	"fmt"
	"go/parser"
	"go/token"
	"strings"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/go-go-golems/ragkit/rag/chunking"
)

func ExtractAll(ctx context.Context, documents []rag.Document) ([]Record, []Diagnostic, error) {
	records := make([]Record, 0)
	diagnostics := make([]Diagnostic, 0)
	extractor := Extractor{}
	for _, document := range documents {
		documentRecords, documentDiagnostics, err := extractor.Extract(ctx, document)
		if err != nil {
			return nil, diagnostics, fmt.Errorf("extract %s: %w", document.SourceURI, err)
		}
		records = append(records, documentRecords...)
		diagnostics = append(diagnostics, documentDiagnostics...)
	}
	return records, diagnostics, nil
}

func FixedAll(ctx context.Context, documents []rag.Document, sizeRunes, overlapRunes int) ([]Record, []Diagnostic, error) {
	chunker := &chunking.Fixed{SizeRunes: sizeRunes, OverlapRunes: overlapRunes}
	records := make([]Record, 0)
	diagnostics := make([]Diagnostic, 0)
	for _, document := range documents {
		chunks, err := chunker.Chunk(ctx, document)
		if err != nil {
			return nil, diagnostics, fmt.Errorf("fixed chunk %s: %w", document.SourceURI, err)
		}
		packageName, diagnostic := packageName(document)
		if diagnostic != nil {
			diagnostics = append(diagnostics, *diagnostic)
		}
		for _, chunk := range chunks {
			lineStart := 1 + strings.Count(document.Text[:chunk.Range.ByteStart], "\n")
			lineEnd := lineStart
			if chunk.Range.ByteEnd > chunk.Range.ByteStart {
				lineEnd += strings.Count(document.Text[chunk.Range.ByteStart:chunk.Range.ByteEnd-1], "\n")
			}
			record := Record{Chunk: chunk, Metadata: Metadata{
				Repository: document.Metadata["repository"], Commit: document.Metadata["commit"],
				Path: document.Metadata["path"], PackageName: packageName,
				PackagePath: document.Metadata["package_path"], Kind: "fixed",
				Test:      strings.HasSuffix(document.Metadata["path"], "_test.go"),
				ByteStart: chunk.Range.ByteStart, ByteEnd: chunk.Range.ByteEnd,
				LineStart: lineStart, LineEnd: lineEnd, ParseStatus: "fixed",
			}}
			if err := record.Validate(document); err != nil {
				return nil, diagnostics, err
			}
			records = append(records, record)
		}
	}
	return records, diagnostics, nil
}

func packageName(document rag.Document) (string, *Diagnostic) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, document.Metadata["path"], document.Text, parser.PackageClauseOnly)
	if file != nil && file.Name != nil {
		return file.Name.Name, nil
	}
	diagnostic := &Diagnostic{DocumentID: document.ID, Path: document.Metadata["path"]}
	if err != nil {
		diagnostic.Message = err.Error()
	} else {
		diagnostic.Message = "Go file has no package clause"
	}
	return "unknown", diagnostic
}
