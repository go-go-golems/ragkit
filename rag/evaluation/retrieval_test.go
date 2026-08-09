package evaluation

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/go-go-golems/ragkit/rag"
)

func TestRetrievalMetrics(t *testing.T) {
	t.Parallel()
	metrics, err := Retrieval(
		rag.Query{ID: "q"},
		[]string{"wrong", "right"},
		[]rag.Judgment{{QueryID: "q", Target: "document", TargetID: "right", Grade: 1}},
		[]int{1, 2},
	)
	if err != nil {
		t.Fatalf("Retrieval() error = %v", err)
	}
	if metrics.MRR != 0.5 || metrics.RecallAt[1] != 0 || metrics.RecallAt[2] != 1 {
		t.Fatalf("metrics = %+v", metrics)
	}
}

func TestRetrievalNDCGRemainsFiniteForLargeGrades(t *testing.T) {
	metrics, err := Retrieval(
		rag.Query{ID: "q"}, []string{"high", "low"},
		[]rag.Judgment{
			{QueryID: "q", Target: "document", TargetID: "high", Grade: 1024},
			{QueryID: "q", Target: "document", TargetID: "low", Grade: 1},
		}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(metrics.NDCGAt[2]) || math.IsInf(metrics.NDCGAt[2], 0) {
		t.Fatalf("NDCG@2 = %v", metrics.NDCGAt[2])
	}
	if _, err := json.Marshal(metrics); err != nil {
		t.Fatalf("marshal metrics: %v", err)
	}
}

func TestRetrievalRejectsInvalidRelevanceGrades(t *testing.T) {
	for _, grade := range []float64{-1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		_, err := Retrieval(
			rag.Query{ID: "q"}, []string{"target"},
			[]rag.Judgment{{QueryID: "q", Target: "document", TargetID: "target", Grade: grade}},
			[]int{1},
		)
		if err == nil {
			t.Fatalf("Retrieval() accepted invalid grade %v", grade)
		}
	}
}

func TestRetrievalRejectsUnlabeledQuery(t *testing.T) {
	t.Parallel()
	if _, err := Retrieval(rag.Query{ID: "q"}, nil, nil, []int{1}); err == nil {
		t.Fatal("Retrieval() error = nil, want missing judgments")
	}
}

func TestRetrievalPrecisionUsesRequestedCutoff(t *testing.T) {
	t.Parallel()
	metrics, err := Retrieval(
		rag.Query{ID: "q"},
		[]string{"right"},
		[]rag.Judgment{{QueryID: "q", Target: "document", TargetID: "right", Grade: 1}},
		[]int{10},
	)
	if err != nil {
		t.Fatalf("Retrieval() error = %v", err)
	}
	if got := metrics.PrecisionAt[10]; got != 0.1 {
		t.Fatalf("Precision@10 = %v, want 0.1", got)
	}
}

func TestEvaluateRankingsAveragesInQueryOrderAndSkips(t *testing.T) {
	t.Parallel()
	queries := []rag.Query{{ID: "first"}, {ID: "unjudged"}, {ID: "disabled"}, {ID: "empty"}}
	judgments := []rag.Judgment{
		{QueryID: "first", Target: "document", TargetID: "right", Grade: 1},
		{QueryID: "disabled", Target: "document", TargetID: "right", Grade: 1},
		{QueryID: "empty", Target: "document", TargetID: "right", Grade: 1},
	}
	report, err := EvaluateRankings(queries, judgments, map[string][]string{
		"first": {"wrong", "right"},
		"empty": {},
	}, []int{1, 10})
	if err != nil {
		t.Fatal(err)
	}
	if report.EvaluatedQueries != 2 || report.SkippedQueries != 2 {
		t.Fatalf("counts = %d/%d", report.EvaluatedQueries, report.SkippedQueries)
	}
	if report.MRR != 0.25 || report.PrecisionAt[10] != 0.05 || report.RecallAt[10] != 0.5 {
		t.Fatalf("report = %+v", report)
	}
	if len(report.PerQuery) != 2 || report.PerQuery[0].QueryID != "first" || report.PerQuery[1].QueryID != "empty" {
		t.Fatalf("per query = %+v", report.PerQuery)
	}
}

func TestEvaluateRankingsEmptySetPreservesCutoffs(t *testing.T) {
	t.Parallel()
	report, err := EvaluateRankings([]rag.Query{{ID: "q"}}, nil, nil, []int{5, 10})
	if err != nil {
		t.Fatal(err)
	}
	if report.EvaluatedQueries != 0 || report.SkippedQueries != 1 {
		t.Fatalf("report = %+v", report)
	}
	if _, ok := report.PrecisionAt[5]; !ok {
		t.Fatal("PrecisionAt[5] is missing")
	}
	if _, ok := report.RecallAt[5]; !ok {
		t.Fatal("RecallAt[5] is missing")
	}
}

func TestRetrievalRejectsDuplicateJudgments(t *testing.T) {
	t.Parallel()
	_, err := Retrieval(
		rag.Query{ID: "q"}, []string{"target"},
		[]rag.Judgment{
			{QueryID: "q", Target: "document", TargetID: "target", Grade: 1},
			{QueryID: "q", Target: "document", TargetID: "target", Grade: 2},
		},
		[]int{1},
	)
	if err == nil {
		t.Fatal("Retrieval() accepted duplicate judgments")
	}
}

func TestRetrievalRejectsInvalidJudgmentIdentity(t *testing.T) {
	t.Parallel()
	tests := []rag.Judgment{
		{QueryID: "q", Target: "", TargetID: "target", Grade: 1},
		{QueryID: "q", Target: "unsupported", TargetID: "target", Grade: 1},
		{QueryID: "q", Target: "document", TargetID: "", Grade: 1},
	}
	for _, judgment := range tests {
		if _, err := Retrieval(rag.Query{ID: "q"}, nil, []rag.Judgment{judgment}, []int{1}); err == nil {
			t.Fatalf("Retrieval() accepted judgment %+v", judgment)
		}
	}
}

func TestRetrievalRejectsDuplicateCutoffs(t *testing.T) {
	t.Parallel()
	_, err := Retrieval(
		rag.Query{ID: "q"}, []string{"target"},
		[]rag.Judgment{{QueryID: "q", Target: "document", TargetID: "target", Grade: 1}},
		[]int{1, 1},
	)
	if err == nil {
		t.Fatal("Retrieval() accepted duplicate cutoffs")
	}
}

func TestEvaluateRankingsRejectsDuplicateQueryIDs(t *testing.T) {
	t.Parallel()
	_, err := EvaluateRankings([]rag.Query{{ID: "q"}, {ID: "q"}}, nil, nil, []int{1})
	if err == nil {
		t.Fatal("EvaluateRankings() error = nil")
	}
}
