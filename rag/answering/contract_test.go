package answering

import (
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func answerEvidence() []rag.Evidence {
	return []rag.Evidence{
		{Chunk: rag.Chunk{ID: "chunk-a", Text: "oak evidence"}},
		{Chunk: rag.Chunk{ID: "chunk-b", Text: "maple evidence"}},
	}
}

func TestParseGroundedAnswerContractTable(t *testing.T) {
	tests := []struct {
		name     string
		raw      string
		valid    bool
		category AnswerFailureCategory
		abstain  bool
	}{
		{"valid", `{"answer":"Oak.","citation_chunk_ids":["chunk-a"],"abstained":false}`, true, AnswerFailureNone, false},
		{"valid abstention", `{"answer":"","citation_chunk_ids":[],"abstained":true}`, true, AnswerFailureNone, true},
		{"malformed", `{`, false, AnswerFailureParse, true},
		{"trailing object", `{"answer":"","citation_chunk_ids":[],"abstained":true} {}`, false, AnswerFailureParse, true},
		{"unknown field", `{"answer":"","citation_chunk_ids":[],"abstained":true,"extra":1}`, false, AnswerFailureParse, true},
		{"missing required field", `{"citation_chunk_ids":[],"abstained":true}`, false, AnswerFailureParse, true},
		{"blank answer", `{"answer":" ","citation_chunk_ids":["chunk-a"],"abstained":false}`, false, AnswerFailureContract, true},
		{"missing citations", `{"answer":"Oak.","citation_chunk_ids":[],"abstained":false}`, false, AnswerFailureContract, true},
		{"unknown citation", `{"answer":"Oak.","citation_chunk_ids":["chunk-x"],"abstained":false}`, false, AnswerFailureContract, true},
		{"duplicate citation", `{"answer":"Oak.","citation_chunk_ids":["chunk-a","chunk-a"],"abstained":false}`, false, AnswerFailureContract, true},
		{"abstention with citation", `{"answer":"","citation_chunk_ids":["chunk-a"],"abstained":true}`, false, AnswerFailureContract, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			answer, contract := ParseGroundedAnswer(test.raw, answerEvidence())
			require.Equal(t, test.valid, contract.Valid)
			require.Equal(t, test.category, contract.FailureCategory)
			require.Equal(t, test.abstain, answer.Abstained)
			if !test.valid {
				require.Empty(t, answer.CitationChunkIDs)
				require.NotEmpty(t, contract.Failures)
			}
		})
	}
}

func TestApplyContextPolicyUsesWholeChunksDeterministically(t *testing.T) {
	evidence := []rag.Evidence{
		{Chunk: rag.Chunk{ID: "a", Text: "1234"}},
		{Chunk: rag.Chunk{ID: "b", Text: "12345"}},
		{Chunk: rag.Chunk{ID: "c", Text: "12"}},
	}
	first := ApplyContextPolicy(evidence, 2, 8)
	second := ApplyContextPolicy(evidence, 2, 8)
	require.Equal(t, first, second)
	require.Equal(t, []string{"a", "c"}, evidenceIDs(first.Evidence))
	require.Equal(t, []string{"b"}, first.OmittedChunkIDs)
	require.Equal(t, 6, first.UsedRunes)
	require.Equal(t, ContextPolicyWholeEvidenceV1, first.Policy)
}
