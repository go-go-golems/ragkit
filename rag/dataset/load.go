// Package dataset loads the two supported experiment input shapes into the
// small, provider-independent RAG domain types.
package dataset

import (
	"encoding/json"
	"os"

	"github.com/go-go-golems/ragkit/digest"
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
	var documents []rag.Document
	if err := json.Unmarshal(data, &documents); err != nil {
		return nil, errors.Wrap(err, "decode corpus")
	}
	if len(documents) == 0 {
		return nil, errors.New("corpus contains no documents")
	}
	return documents, nil
}

func LoadEvaluation(path string) (EvaluationInput, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return EvaluationInput{}, errors.Wrap(err, "read evaluation data")
	}
	var set rag.EvaluationSet
	if err := json.Unmarshal(data, &set); err != nil {
		return EvaluationInput{}, errors.Wrap(err, "decode evaluation data")
	}
	if len(set.Queries) == 0 {
		return EvaluationInput{}, errors.New("evaluation data contains no queries")
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
