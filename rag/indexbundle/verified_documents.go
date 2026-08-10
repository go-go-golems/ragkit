package indexbundle

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/internal/jsonutil"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// LoadVerifiedDocuments loads the exact source-document revision named by an
// index bundle. Unlike Inspect, this is a strict serving API: missing, moved,
// changed, duplicate, or out-of-root corpus data is an error because serving
// stale titles or URLs would misidentify retrieved evidence.
func LoadVerifiedDocuments(ctx context.Context, bundlePath, corpusRoot string) ([]rag.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	manifest, err := loadVerifiedManifest(ctx, bundlePath)
	if err != nil {
		return nil, errors.Wrap(err, "verify index bundle data")
	}
	chunks, err := loadVerifiedChunks(ctx, manifest)
	if err != nil {
		return nil, errors.Wrap(err, "verify index bundle data")
	}
	verified, err := loadVerifiedStoredData(ctx, chunks)
	if err != nil {
		return nil, errors.Wrap(err, "verify index bundle data")
	}
	bundleManifest := verified.manifest
	corpusPath, err := resolveVerifiedCorpusPath(corpusRoot, bundleManifest.CorpusPath)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(corpusPath)
	if err != nil {
		return nil, errors.Wrap(err, "read index bundle source corpus")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	documents, err := jsonutil.DecodeStrict[[]rag.Document](data)
	if err != nil {
		return nil, errors.Wrap(err, "decode index bundle source corpus")
	}
	if len(documents) != bundleManifest.DocumentCount {
		return nil, errors.Errorf("source corpus has %d documents but bundle manifest records %d", len(documents), bundleManifest.DocumentCount)
	}
	seen := make(map[string]struct{}, len(documents))
	for _, document := range documents {
		id := strings.TrimSpace(document.ID)
		if id == "" {
			return nil, errors.New("source corpus contains a document without an ID")
		}
		if _, duplicate := seen[id]; duplicate {
			return nil, errors.Errorf("source corpus contains duplicate document %q", id)
		}
		seen[id] = struct{}{}
	}
	actualDigest, err := digest.JSON(documents)
	if err != nil {
		return nil, errors.Wrap(err, "digest index bundle source corpus")
	}
	if actualDigest != bundleManifest.CorpusDigest {
		return nil, errors.Errorf("source corpus digest %q differs from bundle manifest digest %q", actualDigest, bundleManifest.CorpusDigest)
	}
	if err := rag.ValidateCorpus(documents, verified.chunks); err != nil {
		return nil, errors.Wrap(err, "validate source corpus lineage")
	}
	return append([]rag.Document(nil), documents...), nil
}

func resolveVerifiedCorpusPath(root, candidate string) (string, error) {
	root = strings.TrimSpace(root)
	candidate = strings.TrimSpace(candidate)
	if root == "" {
		return "", errors.New("source corpus root is required")
	}
	if candidate == "" {
		return "", errors.New("bundle manifest records no source corpus path")
	}
	rootPath, err := filepath.Abs(root)
	if err != nil {
		return "", errors.Wrap(err, "resolve source corpus root")
	}
	rootPath, err = filepath.EvalSymlinks(rootPath)
	if err != nil {
		return "", errors.Wrap(err, "evaluate source corpus root")
	}
	corpusPath := candidate
	if !filepath.IsAbs(corpusPath) {
		corpusPath = filepath.Join(rootPath, filepath.Clean(corpusPath))
	}
	corpusPath, err = filepath.EvalSymlinks(corpusPath)
	if err != nil {
		return "", errors.Wrap(err, "evaluate source corpus path")
	}
	relative, err := filepath.Rel(rootPath, corpusPath)
	if err != nil {
		return "", errors.Wrap(err, "compare source corpus path with root")
	}
	if relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", errors.New("bundle source corpus path escapes the configured corpus root")
	}
	return corpusPath, nil
}
