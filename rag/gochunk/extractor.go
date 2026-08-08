package gochunk

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

type Extractor struct{}

func (Extractor) Name() string { return ExtractorVersion }

func (Extractor) Extract(ctx context.Context, document rag.Document) ([]Record, []Diagnostic, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if err := rag.ValidateDocument(document); err != nil {
		return nil, nil, err
	}
	repository := document.Metadata["repository"]
	commit := document.Metadata["commit"]
	path := document.Metadata["path"]
	if repository == "" || commit == "" || path == "" {
		return nil, nil, errors.New("Go document requires repository, commit, and path metadata")
	}

	fset := token.NewFileSet()
	file, parseErr := parser.ParseFile(fset, path, document.Text, parser.ParseComments|parser.AllErrors)
	diagnostics := make([]Diagnostic, 0, 1)
	status := "syntax"
	if parseErr != nil {
		diagnostics = append(diagnostics, Diagnostic{DocumentID: document.ID, Path: path, Message: parseErr.Error()})
		status = "syntax-errors"
	}
	if file == nil {
		return nil, diagnostics, errors.Wrap(parseErr, "parse Go source")
	}

	packageName := file.Name.Name
	packagePath := document.Metadata["package_path"]
	records := make([]Record, 0, len(file.Decls)+1)
	if file.Doc != nil {
		record, err := buildRecord(document, fset, file.Doc.Pos(), file.Name.End(), len(records), Metadata{
			Repository: repository, Commit: commit, Path: path,
			PackageName: packageName, PackagePath: packagePath,
			Kind: "package", Name: packageName, QualifiedName: qualify(packagePath, packageName),
			Documentation: strings.TrimSpace(file.Doc.Text()), ParseStatus: status,
		})
		if err != nil {
			return nil, diagnostics, err
		}
		records = append(records, record)
	}

	for _, declaration := range file.Decls {
		if err := ctx.Err(); err != nil {
			return nil, diagnostics, err
		}
		metadata, start, end, ok, err := declarationMetadata(fset, file, declaration, packagePath, path, repository, commit, status)
		if err != nil {
			return nil, diagnostics, err
		}
		if !ok {
			continue
		}
		startOffset := fset.PositionFor(start, false).Offset
		endOffset := fset.PositionFor(end, false).Offset
		if startOffset < 0 || endOffset < startOffset || endOffset > len(document.Text) {
			diagnostics = append(diagnostics, Diagnostic{
				DocumentID: document.ID, Path: path,
				Message: fmt.Sprintf("skip malformed declaration range [%d,%d)", startOffset, endOffset),
			})
			continue
		}
		record, err := buildRecord(document, fset, start, end, len(records), metadata)
		if err != nil {
			return nil, diagnostics, err
		}
		records = append(records, record)
	}
	return records, diagnostics, nil
}

func declarationMetadata(fset *token.FileSet, file *ast.File, declaration ast.Decl, packagePath, path, repository, commit, status string) (Metadata, token.Pos, token.Pos, bool, error) {
	base := Metadata{
		Repository: repository, Commit: commit, Path: path,
		PackageName: file.Name.Name, PackagePath: packagePath,
		Test: strings.HasSuffix(path, "_test.go"), ParseStatus: status,
	}
	switch node := declaration.(type) {
	case *ast.FuncDecl:
		base.Kind = functionKind(node, base.Test)
		base.Name = node.Name.Name
		base.Exported = ast.IsExported(node.Name.Name)
		base.Documentation = commentText(node.Doc)
		base.Receiver = receiverString(fset, node)
		base.Signature = nodeString(fset, node.Type)
		if base.Receiver != "" {
			base.QualifiedName = qualify(packagePath, "("+base.Receiver+")."+base.Name)
		} else {
			base.QualifiedName = qualify(packagePath, base.Name)
		}
		return base, commentStart(node.Doc, node.Pos()), node.End(), true, nil
	case *ast.GenDecl:
		switch node.Tok {
		case token.TYPE:
			base.Kind = "type"
		case token.CONST:
			base.Kind = "const"
		case token.VAR:
			base.Kind = "var"
		default:
			return Metadata{}, 0, 0, false, nil
		}
		base.Documentation = commentText(node.Doc)
		if len(node.Specs) == 1 {
			base.Name = specName(node.Specs[0])
			base.Exported = ast.IsExported(base.Name)
			base.QualifiedName = qualify(packagePath, base.Name)
		}
		base.Signature = declarationHeader(fset, node)
		return base, commentStart(node.Doc, node.Pos()), node.End(), true, nil
	default:
		return Metadata{}, 0, 0, false, nil
	}
}

func buildRecord(document rag.Document, fset *token.FileSet, startPos, endPos token.Pos, ordinal int, metadata Metadata) (Record, error) {
	start := fset.PositionFor(startPos, false).Offset
	end := fset.PositionFor(endPos, false).Offset
	if start < 0 || end < start || end > len(document.Text) {
		return Record{}, errors.Errorf("invalid Go declaration range [%d,%d) in %s", start, end, metadata.Path)
	}
	text := document.Text[start:end]
	identity, err := digest.JSON(struct {
		DocumentID, Extractor, Kind, QualifiedName, ContentDigest string
		ByteStart, ByteEnd                                        int
	}{document.ID, ExtractorVersion, metadata.Kind, metadata.QualifiedName, digest.Text(text), start, end})
	if err != nil {
		return Record{}, err
	}
	endLinePosition := endPos
	if endPos > startPos {
		endLinePosition--
	}
	metadata.ByteStart = start
	metadata.ByteEnd = end
	metadata.LineStart = fset.PositionFor(startPos, false).Line
	metadata.LineEnd = fset.PositionFor(endLinePosition, false).Line
	record := Record{Chunk: rag.Chunk{
		ID: "chunk-" + identity[:16], DocumentID: document.ID, Ordinal: ordinal,
		Range: rag.Range{ByteStart: start, ByteEnd: end}, Text: text,
		ContentDigest: digest.Text(text), Chunker: ExtractorVersion,
	}, Metadata: metadata}
	if err := record.Validate(document); err != nil {
		return Record{}, err
	}
	return record, nil
}

func functionKind(function *ast.FuncDecl, testFile bool) string {
	if function.Recv != nil {
		return "method"
	}
	if testFile {
		switch {
		case strings.HasPrefix(function.Name.Name, "Example"):
			return "example"
		case strings.HasPrefix(function.Name.Name, "Test"):
			return "test"
		case strings.HasPrefix(function.Name.Name, "Benchmark"):
			return "benchmark"
		case strings.HasPrefix(function.Name.Name, "Fuzz"):
			return "fuzz"
		}
	}
	return "function"
}

func receiverString(fset *token.FileSet, function *ast.FuncDecl) string {
	if function.Recv == nil || len(function.Recv.List) == 0 {
		return ""
	}
	return nodeString(fset, function.Recv.List[0].Type)
}

func nodeString(fset *token.FileSet, node any) string {
	var builder strings.Builder
	if printable, ok := node.(ast.Node); ok {
		_ = printer.Fprint(&builder, fset, printable)
	}
	return builder.String()
}

func declarationHeader(fset *token.FileSet, declaration *ast.GenDecl) string {
	var builder strings.Builder
	_ = printer.Fprint(&builder, fset, declaration)
	value := builder.String()
	if index := strings.Index(value, "\n"); index >= 0 {
		return strings.TrimSpace(value[:index])
	}
	return strings.TrimSpace(value)
}

func specName(spec ast.Spec) string {
	switch node := spec.(type) {
	case *ast.TypeSpec:
		return node.Name.Name
	case *ast.ValueSpec:
		if len(node.Names) == 1 {
			return node.Names[0].Name
		}
	}
	return ""
}

func commentStart(comment *ast.CommentGroup, fallback token.Pos) token.Pos {
	if comment != nil {
		return comment.Pos()
	}
	return fallback
}

func commentText(comment *ast.CommentGroup) string {
	if comment == nil {
		return ""
	}
	return strings.TrimSpace(comment.Text())
}

func qualify(packagePath, name string) string {
	if name == "" {
		return ""
	}
	if packagePath == "" {
		return name
	}
	return packagePath + "." + name
}

func IsGenerated(text string) bool {
	lines := strings.SplitN(text, "\n", 12)
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "// Code generated ") && strings.HasSuffix(trimmed, " DO NOT EDIT.") {
			return true
		}
	}
	return false
}
