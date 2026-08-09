package evaluation

import (
	"sort"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// A query's judgments must all use ONE target level.
//
// `Retrieval` enforces this by erroring, and the error is the problem: a query
// that errors is not a query that scores zero, it is a query that removes
// itself from every metric. `EvaluateRankings` returns the error to its caller,
// so in practice a mixed query fails the whole run — and a caller that logs and
// continues would silently shrink the denominator instead.
//
// The functions here exist so that anything WRITING a judgment can refuse
// before that happens, and so that a tile can say what level a query is judged
// at before offering a grade. They are deliberately in this package, next to
// the error they protect, rather than in whatever server ends up calling them.

// QueryTargetLevel reports the single target level a query's judgments use.
//
// The second return is false when the query has no judgments at all, which is
// not an error: an unjudged query is a coverage hole, `Retrieval` skips it, and
// `queries list --unjudged` already surfaces it. A caller writing the FIRST
// judgment for a query is therefore free to choose the level.
func QueryTargetLevel(queryID string, judgments []rag.Judgment) (Target, bool, error) {
	level := ""
	for _, judgment := range judgments {
		if judgment.QueryID != queryID {
			continue
		}
		if level == "" {
			level = judgment.Target
			continue
		}
		if level != judgment.Target {
			return "", false, errors.Errorf(
				"query %q mixes relevance targets (%q and %q)",
				queryID, level, judgment.Target,
			)
		}
	}
	if level == "" {
		return "", false, nil
	}
	return Target(level), true, nil
}

// ValidateJudgmentTarget refuses a judgment that would mix target levels.
//
// This is the check a write path owes its caller. Without it "grade this chunk"
// against a unit-level query succeeds, writes a valid-looking record, and
// removes that query from every future metric — a failure with no error, no
// warning, and a number that merely looks slightly different.
//
// The message names the existing level because that is the actionable half:
// the caller cannot fix "mixed", it can only grade at the level already in use.
func ValidateJudgmentTarget(queryID string, existing []rag.Judgment, proposed Target) error {
	if proposed == "" {
		return errors.Errorf("judgment for query %q has no target level", queryID)
	}
	if err := rag.Target(proposed).Validate(); err != nil {
		return errors.Wrapf(err, "judgment for query %q has invalid target level", queryID)
	}
	level, judged, err := QueryTargetLevel(queryID, existing)
	if err != nil {
		return err
	}
	if !judged {
		// The first judgment chooses the level for every judgment after it.
		return nil
	}
	if level != proposed {
		return errors.Errorf(
			"query %q is judged at %q; a %q judgment would mix relevance targets "+
				"and remove the query from every metric",
			queryID, level, proposed,
		)
	}
	return nil
}

// MixedTargetQueries lists the queries in a set whose judgments already span
// more than one target level, sorted.
//
// Each one is a query that errors out of `Retrieval` — so this turns a failure
// that is otherwise invisible until a run dies into something a tile or a
// doctor command can show before anyone spends provider budget on it.
func MixedTargetQueries(set rag.EvaluationSet) []string {
	levels := map[string]string{}
	mixed := map[string]struct{}{}
	for _, judgment := range set.Judgments {
		known, seen := levels[judgment.QueryID]
		if !seen {
			levels[judgment.QueryID] = judgment.Target
			continue
		}
		if known != judgment.Target {
			mixed[judgment.QueryID] = struct{}{}
		}
	}
	out := make([]string, 0, len(mixed))
	for queryID := range mixed {
		out = append(out, queryID)
	}
	sort.Strings(out)
	return out
}

// TargetLevels reports the level each query is judged at, and how many
// judgments it carries. Queries with no judgments are absent.
//
// The count is what makes a coverage hole visible: a query judged once is not
// obviously different from an unjudged one in any aggregate, and it is the
// difference between recall@10 measuring something and measuring almost
// nothing.
func TargetLevels(set rag.EvaluationSet) map[string]QueryJudgmentSummary {
	out := map[string]QueryJudgmentSummary{}
	for _, judgment := range set.Judgments {
		summary := out[judgment.QueryID]
		summary.Count++
		if judgment.Grade > 0 {
			summary.Relevant++
		}
		switch {
		case summary.Target == "":
			summary.Target = Target(judgment.Target)
		case string(summary.Target) != judgment.Target:
			summary.Mixed = true
		}
		out[judgment.QueryID] = summary
	}
	return out
}

// QueryJudgmentSummary is one query's judgment coverage.
type QueryJudgmentSummary struct {
	Target Target `json:"target"`
	Count  int    `json:"count"`
	// Relevant counts judgments with a grade above zero, which is what
	// `recall`'s denominator uses. Grade 0 means EXPLICITLY IRRELEVANT and
	// participates in the metrics; absent means never assessed.
	Relevant int `json:"relevant"`
	// Mixed means this query already errors out of Retrieval.
	Mixed bool `json:"mixed,omitempty"`
}
