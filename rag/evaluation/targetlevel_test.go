package evaluation

import (
	"testing"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/stretchr/testify/require"
)

func judgment(queryID, target, targetID string, grade float64) rag.Judgment {
	return rag.Judgment{QueryID: queryID, Target: target, TargetID: targetID, Grade: grade}
}

func TestQueryTargetLevelReportsTheSingleLevelInUse(t *testing.T) {
	judgments := []rag.Judgment{
		judgment("q1", "unit", "unit:aaa", 1),
		judgment("q1", "unit", "unit:bbb", 1),
		judgment("q2", "chunk", "chunk-1", 2),
	}
	level, judged, err := QueryTargetLevel("q1", judgments)
	require.NoError(t, err)
	require.True(t, judged)
	require.Equal(t, TargetUnit, level)

	level, judged, err = QueryTargetLevel("q2", judgments)
	require.NoError(t, err)
	require.True(t, judged)
	require.Equal(t, TargetChunk, level)
}

// An unjudged query is a coverage hole, not an error. Retrieval skips it, and a
// caller writing its FIRST judgment is free to choose the level.
func TestAnUnjudgedQueryHasNoLevelAndIsNotAnError(t *testing.T) {
	level, judged, err := QueryTargetLevel("q-missing", []rag.Judgment{
		judgment("q1", "unit", "unit:aaa", 1),
	})
	require.NoError(t, err)
	require.False(t, judged)
	require.Empty(t, level)
}

func TestQueryTargetLevelReportsAnAlreadyMixedQuery(t *testing.T) {
	_, _, err := QueryTargetLevel("q1", []rag.Judgment{
		judgment("q1", "unit", "unit:aaa", 1),
		judgment("q1", "chunk", "chunk-1", 1),
	})
	require.ErrorContains(t, err, `query "q1" mixes relevance targets`)
	require.ErrorContains(t, err, `"unit"`)
	require.ErrorContains(t, err, `"chunk"`)
}

/*
The check that makes the write path safe.

The live TTC evaluation set judges at `unit` level. Adding a chunk-level
judgment to one of its queries produces a valid-looking record, no error, and a
query that errors out of Retrieval from then on — which removes it from every
metric with nothing to notice.
*/
func TestValidateJudgmentTargetRefusesAMixAndNamesTheExistingLevel(t *testing.T) {
	existing := []rag.Judgment{judgment("ttc-expand-001", "unit", "unit:6cde2a6993744d7f", 1)}

	require.NoError(t, ValidateJudgmentTarget("ttc-expand-001", existing, TargetUnit))

	err := ValidateJudgmentTarget("ttc-expand-001", existing, TargetChunk)
	require.Error(t, err)
	require.ErrorContains(t, err, `is judged at "unit"`)
	require.ErrorContains(t, err, "remove the query from every metric")
}

func TestValidateJudgmentTargetAcceptsTheFirstJudgmentAtAnyLevel(t *testing.T) {
	for _, target := range []Target{TargetRepresentation, TargetChunk, TargetDocument, TargetUnit} {
		require.NoError(t, ValidateJudgmentTarget("fresh", nil, target))
	}
}

func TestValidateJudgmentTargetRejectsAnUnknownOrMissingLevel(t *testing.T) {
	require.ErrorContains(t, ValidateJudgmentTarget("q1", nil, ""), "no target level")
	require.ErrorContains(t,
		ValidateJudgmentTarget("q1", nil, Target("passage")),
		`unsupported relevance target "passage"`,
	)
}

// A mixed query is invisible until a run dies on it. This is what lets a tile
// or a doctor command show it first.
func TestMixedTargetQueriesFindsThemBeforeARunDoes(t *testing.T) {
	set := rag.EvaluationSet{Judgments: []rag.Judgment{
		judgment("clean", "unit", "unit:aaa", 1),
		judgment("clean", "unit", "unit:bbb", 1),
		judgment("broken", "unit", "unit:ccc", 1),
		judgment("broken", "chunk", "chunk-9", 1),
		judgment("also-broken", "document", "doc-1", 1),
		judgment("also-broken", "unit", "unit:ddd", 1),
	}}
	require.Equal(t, []string{"also-broken", "broken"}, MixedTargetQueries(set))
}

func TestTargetLevelsReportsCoveragePerQuery(t *testing.T) {
	set := rag.EvaluationSet{Judgments: []rag.Judgment{
		judgment("q1", "unit", "unit:aaa", 1),
		judgment("q1", "unit", "unit:bbb", 0),
		judgment("q2", "chunk", "chunk-1", 3),
		judgment("q3", "unit", "unit:ccc", 1),
		judgment("q3", "chunk", "chunk-2", 1),
	}}
	levels := TargetLevels(set)

	// Grade 0 is EXPLICITLY IRRELEVANT and participates in the metrics; only
	// grades above zero count toward recall's denominator.
	require.Equal(t, QueryJudgmentSummary{Target: TargetUnit, Count: 2, Relevant: 1}, levels["q1"])
	require.Equal(t, QueryJudgmentSummary{Target: TargetChunk, Count: 1, Relevant: 1}, levels["q2"])
	require.True(t, levels["q3"].Mixed)
	require.NotContains(t, levels, "q-unjudged")
}

// The invariant these functions exist to protect, asserted against the real
// metrics code rather than restated.
func TestRetrievalStillErrorsOnTheMixValidateWouldHaveRefused(t *testing.T) {
	query := rag.Query{ID: "q1", Text: "when to prune?"}
	mixed := []rag.Judgment{
		judgment("q1", "unit", "unit:aaa", 1),
		judgment("q1", "chunk", "chunk-1", 1),
	}
	require.Error(t, ValidateJudgmentTarget("q1", mixed[:1], TargetChunk))

	_, err := Retrieval(query, []string{"unit:aaa"}, mixed, []int{10})
	require.ErrorContains(t, err, "mixes relevance targets")
}
