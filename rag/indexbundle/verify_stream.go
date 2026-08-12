package indexbundle

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

type streamedChunks struct {
	verifiedManifest
	chunkDigest string
	relation    *verificationRelation
}

// streamJSONArray decodes a strict JSON array one element at a time and
// computes the same digest as encoding/json over a non-nil slice.
func streamJSONArray[T any](ctx context.Context, path string, consume func(int, T) error) (int, string, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, "", errors.Wrap(err, "open JSON array")
	}
	defer func() { _ = file.Close() }()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	token, err := decoder.Token()
	if err != nil {
		return 0, "", errors.Wrap(err, "read JSON array start")
	}
	if delimiter, ok := token.(json.Delim); !ok || delimiter != '[' {
		return 0, "", errors.New("JSON input must be an array")
	}
	count := 0
	contentDigest, err := digest.JSONSequence(ctx, func(yield func(T) error) error {
		for decoder.More() {
			if err := ctx.Err(); err != nil {
				return err
			}
			var value T
			if err := decoder.Decode(&value); err != nil {
				return errors.Wrapf(err, "decode JSON array element %d", count)
			}
			if consume != nil {
				if err := consume(count, value); err != nil {
					return err
				}
			}
			if err := yield(value); err != nil {
				return err
			}
			count++
		}
		end, err := decoder.Token()
		if err != nil {
			return errors.Wrap(err, "read JSON array end")
		}
		if delimiter, ok := end.(json.Delim); !ok || delimiter != ']' {
			return errors.New("JSON input has no array terminator")
		}
		var trailing any
		if err := decoder.Decode(&trailing); !stderrors.Is(err, io.EOF) {
			if err == nil {
				return errors.New("JSON input contains trailing data")
			}
			return errors.Wrap(err, "decode trailing JSON data")
		}
		return nil
	})
	if err != nil {
		return count, "", err
	}
	return count, contentDigest, nil
}

func streamVerifiedChunks(ctx context.Context, state verifiedManifest, relation *verificationRelation) (streamedChunks, error) {
	count, chunkDigest, err := streamJSONArray[rag.Chunk](ctx, filepath.Join(state.path, chunksName), func(_ int, chunk rag.Chunk) error {
		if chunk.ID == "" || chunk.DocumentID == "" || chunk.ContentDigest == "" {
			return errors.New("bundle contains an invalid chunk identity")
		}
		if digest.Text(chunk.Text) != chunk.ContentDigest {
			return errors.Errorf("bundle chunk %q content digest mismatch", chunk.ID)
		}
		if chunk.Range.ByteStart < 0 || chunk.Range.ByteEnd < chunk.Range.ByteStart || chunk.Range.ByteEnd-chunk.Range.ByteStart != len([]byte(chunk.Text)) {
			return errors.Errorf("bundle chunk %q has an invalid stored byte range", chunk.ID)
		}
		if chunk.Ordinal < 0 {
			return errors.Errorf("bundle chunk %q has negative ordinal %d", chunk.ID, chunk.Ordinal)
		}
		return relation.addChunk(ctx, chunk.ID, chunk.DocumentID, chunk.Ordinal, chunk.ContentDigest)
	})
	if err != nil {
		return streamedChunks{}, errors.Wrap(err, "stream bundle chunks")
	}
	if err := relation.finishChunks(ctx); err != nil {
		return streamedChunks{}, err
	}
	if count != state.manifest.ChunkCount {
		return streamedChunks{}, errors.Errorf("bundle data counts differ from manifest: holds %d chunks but manifest counts %d", count, state.manifest.ChunkCount)
	}
	documentCount, err := relation.documentCount(ctx)
	if err != nil {
		return streamedChunks{}, err
	}
	if documentCount > state.manifest.DocumentCount {
		return streamedChunks{}, errors.New("bundle chunk document count exceeds manifest corpus count")
	}
	if chunkDigest != state.manifest.ChunkDigest {
		return streamedChunks{}, errors.New("bundle chunk digest differs from manifest")
	}
	return streamedChunks{verifiedManifest: state, chunkDigest: chunkDigest, relation: relation}, nil
}

func streamVerifiedStoredIdentity(ctx context.Context, state streamedChunks) (verifiedManifest, error) {
	kindSet := make(map[string]struct{}, len(state.manifest.RepresentationKinds))
	count, representationDigest, err := streamJSONArray[rag.Representation](ctx, filepath.Join(state.path, representationsName), func(index int, representation rag.Representation) error {
		if strings.TrimSpace(representation.ID) == "" {
			return errors.Errorf("representation %d has no ID", index)
		}
		if err := state.relation.addRepresentation(ctx, representation.ID); err != nil {
			return err
		}
		chunkContentDigest, exists, err := state.relation.chunkContentDigest(ctx, representation.ChunkID)
		if err != nil {
			return err
		}
		if !exists {
			return errors.Errorf("representation %q references unknown chunk %q", representation.ID, representation.ChunkID)
		}
		if strings.TrimSpace(representation.Kind) == "" {
			return errors.Errorf("representation %q has no kind", representation.ID)
		}
		if !utf8.ValidString(representation.Text) {
			return errors.Errorf("representation %q contains invalid UTF-8", representation.ID)
		}
		if representation.ContentDigest == "" {
			return errors.Errorf("representation %q content digest is required", representation.ID)
		}
		if actual := digest.Text(representation.Text); actual != representation.ContentDigest {
			return errors.Errorf("representation %q content digest mismatch: stored=%s actual=%s", representation.ID, representation.ContentDigest, actual)
		}
		if representation.Kind == "raw" && representation.ContentDigest != chunkContentDigest {
			return errors.Errorf("raw representation %q differs from chunk %q", representation.ID, representation.ChunkID)
		}
		kindSet[representation.Kind] = struct{}{}
		return nil
	})
	if err != nil {
		return verifiedManifest{}, errors.Wrap(err, "stream bundle representations")
	}
	if err := state.relation.finishRepresentations(ctx); err != nil {
		return verifiedManifest{}, err
	}
	if count != state.manifest.RepresentationCount {
		return verifiedManifest{}, errors.New("bundle representation count differs from manifest")
	}
	kinds := make([]string, 0, len(kindSet))
	for kind := range kindSet {
		kinds = append(kinds, kind)
	}
	sort.Strings(kinds)
	expectedID, err := calculateIDFromDigests(state.manifest.CorpusDigest, state.chunkDigest, representationDigest, kinds, state.manifest.Chunker, state.manifest.Lexical, state.manifest.Vector, state.manifest.Content)
	if err != nil {
		return verifiedManifest{}, err
	}
	if expectedID != state.manifest.BundleID || !reflect.DeepEqual(kinds, state.manifest.RepresentationKinds) {
		return verifiedManifest{}, errors.New("bundle identity does not match stored data")
	}
	if state.manifest.Vector != nil && state.manifest.Vector.RepresentationDigest != representationDigest {
		return verifiedManifest{}, errors.New("bundle vector representation digest mismatch")
	}
	return state.verifiedManifest, nil
}
