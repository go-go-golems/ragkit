// Package dataset loads the two supported experiment input shapes into the
// small, provider-independent RAG domain types.
package dataset

import (
	"os"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/internal/jsonutil"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

type EvaluationInput struct {
	CorpusDigest string
	Queries      []rag.Query
	Judgments    []rag.Judgment
}

func LoadDocuments(path string) ([]rag.Document, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, errors.Wrap(err, "read corpus")
	}
	documents, err := jsonutil.DecodeStrict[[]rag.Document](data)
	if err != nil {
		return nil, errors.Wrap(err, "decode corpus")
	}
	if len(documents) == 0 {
		return nil, errors.New("corpus contains no documents")
	}
	if err := rag.ValidateCorpus(documents, nil); err != nil {
		return nil, errors.Wrap(err, "validate corpus")
	}
	return documents, nil
}

func LoadEvaluation(path string) (EvaluationInput, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return EvaluationInput{}, errors.Wrap(err, "read evaluation data")
	}
	set, err := jsonutil.DecodeStrict[rag.EvaluationSet](data)
	if err != nil {
		return EvaluationInput{}, errors.Wrap(err, "decode evaluation data")
	}
	if len(set.Queries) == 0 {
		return EvaluationInput{}, errors.New("evaluation data contains no queries")
	}
	if err := rag.ValidateQueries(set.Queries); err != nil {
		return EvaluationInput{}, errors.Wrap(err, "validate evaluation queries")
	}
	queryIDs := make(map[string]struct{}, len(set.Queries))
	for _, query := range set.Queries {
		queryIDs[query.ID] = struct{}{}
	}
	for index, judgment := range set.Judgments {
		if _, ok := queryIDs[judgment.QueryID]; !ok {
			return EvaluationInput{}, errors.Errorf("judgment %d refers to unknown query %q", index, judgment.QueryID)
		}
	}
	return EvaluationInput{
		CorpusDigest: set.CorpusDigest, Queries: set.Queries, Judgments: set.Judgments,
	}, nil
}

func ValidateCorpus(documents []rag.Document, evaluation EvaluationInput) error {
	if evaluation.CorpusDigest == "" {
		return nil
	}
	actual, err := digest.JSON(documents)
	if err != nil {
		return errors.Wrap(err, "digest corpus")
	}
	if actual != evaluation.CorpusDigest {
		return errors.Errorf(
			"evaluation corpus digest %q does not match loaded corpus digest %q",
			evaluation.CorpusDigest, actual,
		)
	}
	return nil
}
