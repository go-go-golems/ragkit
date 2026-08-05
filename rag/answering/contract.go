package answering

import (
	"strings"

	"github.com/go-go-golems/ragkit/internal/jsonutil"
	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// stripCodeFence removes one markdown code fence wrapping the whole payload.
// Several models wrap their JSON in ```json fences no matter what the prompt
// says. The fence is transport, not content: everything inside still decodes
// strictly, so the contract loses nothing.
func stripCodeFence(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if !strings.HasPrefix(trimmed, "```") {
		return raw
	}
	firstLineEnd := strings.IndexByte(trimmed, '\n')
	if firstLineEnd < 0 || !strings.HasSuffix(trimmed, "```") {
		return raw
	}
	inner := trimmed[firstLineEnd+1 : len(trimmed)-3]
	return strings.TrimSpace(inner)
}

// ParseGroundedAnswer strictly decodes and validates one provider response.
// Invalid output becomes a safe abstention while diagnostics remain available
// in the returned contract result.
func ParseGroundedAnswer(raw string, evidence []rag.Evidence) (GroundedAnswer, AnswerContractResult) {
	var wire struct {
		Text             *string   `json:"answer"`
		CitationChunkIDs *[]string `json:"citation_chunk_ids"`
		Abstained        *bool     `json:"abstained"`
	}
	if err := jsonutil.DecodeStrictInto([]byte(stripCodeFence(raw)), &wire); err != nil {
		return safeAbstention(), AnswerContractResult{
			FailureCategory: AnswerFailureParse,
			Failures:        []string{errors.Wrap(err, "decode grounded answer").Error()},
		}
	}
	missing := make([]string, 0, 3)
	if wire.Text == nil {
		missing = append(missing, "answer")
	}
	if wire.CitationChunkIDs == nil {
		missing = append(missing, "citation_chunk_ids")
	}
	if wire.Abstained == nil {
		missing = append(missing, "abstained")
	}
	if len(missing) > 0 {
		return safeAbstention(), AnswerContractResult{
			FailureCategory: AnswerFailureParse,
			Failures:        []string{"grounded answer is missing required fields: " + strings.Join(missing, ", ")},
		}
	}
	answer := GroundedAnswer{
		Text:             *wire.Text,
		CitationChunkIDs: *wire.CitationChunkIDs,
		Abstained:        *wire.Abstained,
	}
	contract := ValidateGroundedAnswer(answer, evidence)
	if !contract.Valid {
		return safeAbstention(), contract
	}
	return answer, contract
}

// ValidateGroundedAnswer enforces abstention and citation invariants using
// only the evidence supplied to generation.
func ValidateGroundedAnswer(answer GroundedAnswer, evidence []rag.Evidence) AnswerContractResult {
	failures := make([]string, 0)
	if answer.Abstained {
		if len(answer.CitationChunkIDs) != 0 {
			failures = append(failures, "abstained answer must not contain citations")
		}
	} else {
		if strings.TrimSpace(answer.Text) == "" {
			failures = append(failures, "non-abstained answer must contain text")
		}
		if len(answer.CitationChunkIDs) == 0 {
			failures = append(failures, "non-abstained answer must contain citations")
		}
	}
	allowed := make(map[string]struct{}, len(evidence))
	for _, item := range evidence {
		allowed[item.Chunk.ID] = struct{}{}
	}
	seen := make(map[string]struct{}, len(answer.CitationChunkIDs))
	for _, citation := range answer.CitationChunkIDs {
		if _, duplicate := seen[citation]; duplicate {
			failures = append(failures, "duplicate citation chunk ID "+citation)
			continue
		}
		seen[citation] = struct{}{}
		if _, ok := allowed[citation]; !ok {
			failures = append(failures, "unknown citation chunk ID "+citation)
		}
	}
	if len(failures) != 0 {
		return AnswerContractResult{
			FailureCategory: AnswerFailureContract,
			Failures:        failures,
		}
	}
	return AnswerContractResult{Valid: true}
}

func safeAbstention() GroundedAnswer {
	return GroundedAnswer{CitationChunkIDs: []string{}, Abstained: true}
}
