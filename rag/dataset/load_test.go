package dataset

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/rag"
)

// writeFixture builds a small self-contained corpus + evaluation pair in a
// temp dir, computing the digests the loader validates against.
func writeFixture(t *testing.T) (corpusPath, evaluationPath string) {
	t.Helper()
	dir := t.TempDir()

	documents := []rag.Document{
		{
			ID:        "doc-alpha",
			SourceURI: "fixture://doc-alpha",
			Title:     "Alpha Guide",
			Text:      "# Overview\nAlpha widgets require quarterly calibration.\n\n# Storage\nKeep alpha widgets below 30 degrees.",
			Metadata:  map[string]string{"fixture": "true"},
		},
		{
			ID:        "doc-beta",
			SourceURI: "fixture://doc-beta",
			Title:     "Beta Policy",
			Text:      "# Returns\nBeta units may be returned within 30 days with proof of purchase.",
			Metadata:  map[string]string{"fixture": "true"},
		},
	}
	for i := range documents {
		documents[i].ContentDigest = digest.Text(documents[i].Text)
	}
	corpusDigest, err := digest.JSON(documents)
	require.NoError(t, err)

	evaluation := rag.EvaluationSet{
		ID:           "fixture-eval-v1",
		CorpusDigest: corpusDigest,
		Queries: []rag.Query{
			{ID: "q-calibration", Text: "How often do alpha widgets need calibration?", Split: "fixture", Category: "care"},
			{ID: "q-returns", Text: "What is the beta return window?", Split: "fixture", Category: "policy"},
		},
		Judgments: []rag.Judgment{
			{QueryID: "q-calibration", Target: "document", TargetID: "doc-alpha", Grade: 2},
			{QueryID: "q-returns", Target: "document", TargetID: "doc-beta", Grade: 2},
		},
	}

	corpusPath = filepath.Join(dir, "corpus.json")
	evaluationPath = filepath.Join(dir, "evaluation.json")
	corpusJSON, err := json.Marshal(documents)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(corpusPath, corpusJSON, 0o644))
	evaluationJSON, err := json.Marshal(evaluation)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(evaluationPath, evaluationJSON, 0o644))
	return corpusPath, evaluationPath
}

func TestLoadAndValidateFixtureDataset(t *testing.T) {
	corpusPath, evaluationPath := writeFixture(t)

	documents, err := LoadDocuments(corpusPath)
	require.NoError(t, err)
	require.Len(t, documents, 2)

	evaluation, err := LoadEvaluation(evaluationPath)
	require.NoError(t, err)
	require.Len(t, evaluation.Queries, 2)
	require.Len(t, evaluation.Judgments, 2)
	require.NoError(t, ValidateCorpus(documents, evaluation))
}

func TestValidateCorpusRejectsDigestMismatch(t *testing.T) {
	corpusPath, evaluationPath := writeFixture(t)

	documents, err := LoadDocuments(corpusPath)
	require.NoError(t, err)
	documents[0].Text = "tampered"

	evaluation, err := LoadEvaluation(evaluationPath)
	require.NoError(t, err)
	require.Error(t, ValidateCorpus(documents, evaluation))
}

func TestLoadDocumentsRejectsEmptyCorpus(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "corpus.json")
	require.NoError(t, os.WriteFile(path, []byte("[]"), 0o644))
	_, err := LoadDocuments(path)
	require.Error(t, err)
}

func TestLoadDocumentsRejectsInvalidAndDuplicateDocuments(t *testing.T) {
	for _, documents := range [][]rag.Document{
		{{ID: "doc", Text: "body", ContentDigest: "wrong"}},
		{
			{ID: "doc", Text: "one", ContentDigest: digest.Text("one")},
			{ID: "doc", Text: "two", ContentDigest: digest.Text("two")},
		},
	} {
		path := filepath.Join(t.TempDir(), "corpus.json")
		data, err := json.Marshal(documents)
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(path, data, 0o644))
		_, err = LoadDocuments(path)
		require.Error(t, err)
	}
}

func TestLoadEvaluationRejectsOrphanJudgment(t *testing.T) {
	set := rag.EvaluationSet{
		Queries:   []rag.Query{{ID: "known", Text: "question"}},
		Judgments: []rag.Judgment{{QueryID: "missing", Target: "document", TargetID: "doc", Grade: 1}},
	}
	path := filepath.Join(t.TempDir(), "evaluation.json")
	data, err := json.Marshal(set)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(path, data, 0o644))
	_, err = LoadEvaluation(path)
	require.ErrorContains(t, err, "unknown query")
}
