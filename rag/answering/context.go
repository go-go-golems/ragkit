package answering

import (
	"unicode/utf8"

	"github.com/go-go-golems/ragkit/rag"
)

// ApplyContextPolicy admits complete chunks in evidence order. A chunk that
// exceeds the remaining rune budget is omitted rather than truncated.
func ApplyContextPolicy(evidence []rag.Evidence, maximumEvidence, maximumRunes int) ContextPolicyResult {
	result := ContextPolicyResult{
		MaxEvidence: maximumEvidence,
		MaxRunes:    maximumRunes,
		Policy:      ContextPolicyWholeEvidenceV1,
	}
	for _, item := range evidence {
		runes := utf8.RuneCountInString(item.Chunk.Text)
		if len(result.Evidence) >= maximumEvidence || result.UsedRunes+runes > maximumRunes {
			result.OmittedChunkIDs = append(result.OmittedChunkIDs, item.Chunk.ID)
			continue
		}
		result.Evidence = append(result.Evidence, item)
		result.UsedRunes += runes
	}
	return result
}
