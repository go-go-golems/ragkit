package indexbundle

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// CorpusState says what happened when the reader tried to name the documents.
//
// It is a value on the Inspection rather than an error return because a corpus
// that is missing or has moved on must degrade the view, not refuse it. The
// chunks, their byte ranges and their text are all in the bundle; only the
// titles come from outside it, and a screen that shows digests is still a
// useful screen.
type CorpusState string

const (
	// CorpusMatches means the corpus was read and its digest is the one the
	// bundle was built from. Titles are trustworthy.
	CorpusMatches CorpusState = "matches"
	// CorpusDiffers means the corpus was read and its digest is not the one the
	// bundle was built from. The file has changed since the build, so a title
	// may belong to different text than the chunks hold.
	CorpusDiffers CorpusState = "differs"
	// CorpusAbsent means the corpus could not be read at all.
	CorpusAbsent CorpusState = "absent"
)

// Corpus is the outcome of the document join, including the reason when it did
// not fully succeed.
//
// Reason is prose for a person. It is populated for every state except
// CorpusMatches, and a screen that shows a degraded state must show it: a
// missing title with no stated reason reads as a bug in the reader.
type Corpus struct {
	State CorpusState
	// Path is where the reader looked. It is the manifest's CorpusPath resolved
	// against the working directory.
	Path string
	// ExpectedDigest is what the manifest recorded at build time.
	ExpectedDigest string
	// ActualDigest is what the file on disk hashes to now. It is empty when the
	// state is CorpusAbsent.
	ActualDigest string
	Reason       string
}

// Document is one source document with the chunk counts derived from the
// bundle. Title and SourceURI are empty when the corpus did not resolve.
type Document struct {
	ID        string
	Title     string
	SourceURI string
	// Bytes is the byte length of the source text. It is zero when the corpus
	// did not resolve.
	//
	// Bytes and Runes are both needed. A chunk range is in bytes and a chunk
	// size limit is in runes, so a screen that draws cut boundaries needs the
	// byte total and a screen that reports chunk sizes needs the rune total.
	Bytes int
	// Runes is the length of the source text, not the sum of its chunks.
	// Overlap makes the sum of chunk lengths larger than the document, so the
	// two are different measurements and must not be confused. It is zero when
	// the corpus did not resolve.
	Runes int
	// ChunkCount and the rune extremes come from the bundle and are always
	// available, corpus or no corpus.
	ChunkCount     int
	TotalRunes     int
	MinimumRunes   int
	MaximumRunes   int
	FirstOrdinal   int
	LastOrdinal    int
	firstSeenIndex int
}

// Inspection is a bundle opened for reading. It holds no index handle and needs
// no provider.
//
// This is the read side of the same split pkg/session already has: Recorder
// writes and Open reads, and neither knows about the other. A reader that opens
// no index cannot corrupt one, and it cannot fail for want of an embedding
// provider it will never call.
type Inspection struct {
	Path      string
	Manifest  Manifest
	Chunks    []rag.Chunk
	Documents []Document
	Corpus    Corpus
}

// Inspect reads a bundle for inspection. It reads manifest.json and chunks.json
// and nothing else inside the bundle: not representations.json, not
// vectors.sqlite, and not the bleve directory.
//
// The manifest chunk count is checked, because a chunks.json that disagrees
// with its manifest means one of the two files was written by something other
// than Build, and every count a caller derives from the chunks would be
// silently wrong.
func Inspect(ctx context.Context, path string) (*Inspection, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	manifest, err := LoadManifest(path)
	if err != nil {
		return nil, errors.Wrap(err, "load index bundle manifest")
	}
	var chunks []rag.Chunk
	if err := readJSON(filepath.Join(path, chunksName), &chunks); err != nil {
		return nil, errors.Wrap(err, "load bundle chunks")
	}
	if len(chunks) != manifest.ChunkCount {
		return nil, errors.Errorf(
			"bundle holds %d chunks but its manifest counts %d",
			len(chunks), manifest.ChunkCount,
		)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	corpus, documents := readCorpus(manifest)
	return &Inspection{
		Path: path, Manifest: manifest, Chunks: chunks,
		Documents: joinDocuments(chunks, documents), Corpus: corpus,
	}, nil
}

// readCorpus resolves the corpus named by the manifest. It never returns an
// error: every failure is a CorpusState with a stated reason.
func readCorpus(manifest Manifest) (Corpus, map[string]rag.Document) {
	corpus := Corpus{
		Path: manifest.CorpusPath, ExpectedDigest: manifest.CorpusDigest,
	}
	if corpus.Path == "" {
		corpus.State = CorpusAbsent
		corpus.Reason = "the manifest records no corpus path"
		return corpus, nil
	}
	data, err := os.ReadFile(corpus.Path)
	if err != nil {
		corpus.State = CorpusAbsent
		corpus.Reason = "cannot read " + corpus.Path + ": " + err.Error()
		// The manifest stores the corpus path as it was given at build time,
		// which is relative to the repository root. Inspect resolves it against
		// the working directory, so a caller started elsewhere gets a
		// not-found for a file that exists. Naming the directory makes that
		// diagnosable instead of looking like a deleted corpus.
		if !filepath.IsAbs(corpus.Path) {
			if working, wdErr := os.Getwd(); wdErr == nil {
				corpus.Reason += " (resolved against " + working + ")"
			}
		}
		return corpus, nil
	}
	var documents []rag.Document
	if err := json.Unmarshal(data, &documents); err != nil {
		corpus.State = CorpusAbsent
		corpus.Reason = "cannot decode " + corpus.Path + ": " + err.Error()
		return corpus, nil
	}
	// The digest is over the decoded documents, not over the file bytes,
	// because that is what CalculateID does. Hashing the bytes here would
	// report a mismatch for a corpus that was only reformatted.
	actual, err := digest.JSON(documents)
	if err != nil {
		corpus.State = CorpusAbsent
		corpus.Reason = "cannot digest " + corpus.Path + ": " + err.Error()
		return corpus, nil
	}
	corpus.ActualDigest = actual
	byID := make(map[string]rag.Document, len(documents))
	for _, document := range documents {
		byID[document.ID] = document
	}
	if actual != manifest.CorpusDigest {
		corpus.State = CorpusDiffers
		corpus.Reason = corpus.Path +
			" has changed since the bundle was built; a title may not describe" +
			" the text the chunks hold"
		return corpus, byID
	}
	corpus.State = CorpusMatches
	return corpus, byID
}

// joinDocuments rolls the chunks up per document and attaches whatever the
// corpus supplied.
//
// The result is ordered by first appearance in chunks.json rather than by any
// count, so the order is a property of the bundle and does not change when a
// caller sorts by something else. Callers that want a ranking sort explicitly.
func joinDocuments(chunks []rag.Chunk, corpus map[string]rag.Document) []Document {
	byID := make(map[string]*Document, len(corpus))
	order := make([]*Document, 0, len(corpus))
	for index, chunk := range chunks {
		document, seen := byID[chunk.DocumentID]
		if !seen {
			document = &Document{
				ID: chunk.DocumentID, firstSeenIndex: index,
				FirstOrdinal: chunk.Ordinal, LastOrdinal: chunk.Ordinal,
			}
			if source, ok := corpus[chunk.DocumentID]; ok {
				document.Title = source.Title
				document.SourceURI = source.SourceURI
				document.Runes = len([]rune(source.Text))
				document.Bytes = len(source.Text)
			}
			byID[chunk.DocumentID] = document
			order = append(order, document)
		}
		runes := len([]rune(chunk.Text))
		document.ChunkCount++
		document.TotalRunes += runes
		if runes < document.MinimumRunes || document.MinimumRunes == 0 {
			document.MinimumRunes = runes
		}
		if runes > document.MaximumRunes {
			document.MaximumRunes = runes
		}
		if chunk.Ordinal < document.FirstOrdinal {
			document.FirstOrdinal = chunk.Ordinal
		}
		if chunk.Ordinal > document.LastOrdinal {
			document.LastOrdinal = chunk.Ordinal
		}
	}
	sort.SliceStable(order, func(i, j int) bool {
		return order[i].firstSeenIndex < order[j].firstSeenIndex
	})
	documents := make([]Document, 0, len(order))
	for _, document := range order {
		documents = append(documents, *document)
	}
	return documents
}

// TitlesAvailable reports whether a screen can show names instead of digests.
// A differing corpus still supplies titles, and the Corpus reason says they may
// be stale; refusing to show them would lose more than it protects.
func (i *Inspection) TitlesAvailable() bool {
	return i != nil && i.Corpus.State != CorpusAbsent
}

// DocumentByID returns the rollup for one document.
func (i *Inspection) DocumentByID(id string) (Document, bool) {
	if i == nil {
		return Document{}, false
	}
	for _, document := range i.Documents {
		if document.ID == id {
			return document, true
		}
	}
	return Document{}, false
}

// ChunksOf returns the chunks of one document in ordinal order.
//
// It sorts rather than trusting the file order. chunks.json is written in
// document order today, and a screen that draws cut boundaries would show the
// overlap markers against the wrong neighbours if that ever stopped being true.
func (i *Inspection) ChunksOf(documentID string) []rag.Chunk {
	if i == nil {
		return nil
	}
	var chunks []rag.Chunk
	for _, chunk := range i.Chunks {
		if chunk.DocumentID == documentID {
			chunks = append(chunks, chunk)
		}
	}
	sort.SliceStable(chunks, func(a, b int) bool {
		return chunks[a].Ordinal < chunks[b].Ordinal
	})
	return chunks
}

// Name returns a title when one is available and the document digest when one
// is not. Every screen that names a document uses this, so no screen invents
// its own fallback.
func (d Document) Name() string {
	if d.Title != "" {
		return d.Title
	}
	return d.ID
}
