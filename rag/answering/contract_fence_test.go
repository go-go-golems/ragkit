package answering

import (
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func TestParseGroundedAnswerStripsOneCodeFence(t *testing.T) {
	evidence := []rag.Evidence{{Chunk: rag.Chunk{ID: "chunk-a"}}}
	fenced := "```json\n{\"answer\":\"three feet apart\",\"citation_chunk_ids\":[\"chunk-a\"],\"abstained\":false}\n```"
	answer, contract := ParseGroundedAnswer(fenced, evidence)
	require.True(t, contract.Valid)
	require.Equal(t, "three feet apart", answer.Text)

	// A fence that does not wrap the whole payload stays untouched and the
	// strict decoder still refuses it.
	_, broken := ParseGroundedAnswer("```json\n{\"answer\":\"x\"}", evidence)
	require.Equal(t, AnswerFailureParse, broken.FailureCategory)
}
