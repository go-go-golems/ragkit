package evaluation

import (
	"fmt"
	"math"
	"sort"

	"github.com/go-go-golems/ragkit/rag"
)

// RetrievalMetrics contains per-query values before aggregation.
type RetrievalMetrics struct {
	QueryID     string          `json:"query_id"`
	Target      string          `json:"target"`
	PrecisionAt map[int]float64 `json:"precision_at"`
	RecallAt    map[int]float64 `json:"recall_at"`
	HitRateAt   map[int]float64 `json:"hit_rate_at"`
	MRR         float64         `json:"mrr"`
	NDCGAt      map[int]float64 `json:"ndcg_at"`
}

// Report aggregates retrieval metrics over evaluated queries. Queries without
// judgments or without an entry in rankedIDs are skipped. A present empty
// ranking is evaluated as an empty result.
type Report struct {
	EvaluatedQueries int                `json:"evaluated_queries"`
	SkippedQueries   int                `json:"skipped_queries"`
	MRR              float64            `json:"mrr"`
	PrecisionAt      map[int]float64    `json:"precision_at"`
	RecallAt         map[int]float64    `json:"recall_at"`
	NDCGAt           map[int]float64    `json:"ndcg_at"`
	HitRateAt        map[int]float64    `json:"hit_rate_at"`
	PerQuery         []RetrievalMetrics `json:"per_query"`
}

// EvaluateRankings evaluates and averages ranked target identities in query
// order.
func EvaluateRankings(
	queries []rag.Query,
	judgments []rag.Judgment,
	rankedIDs map[string][]string,
	cutoffs []int,
) (Report, error) {
	if err := rag.ValidateQueries(queries); err != nil {
		return Report{}, err
	}
	if err := rag.ValidateJudgments(judgments); err != nil {
		return Report{}, err
	}
	if err := validateCutoffs(cutoffs); err != nil {
		return Report{}, err
	}
	report := Report{
		PrecisionAt: make(map[int]float64, len(cutoffs)),
		RecallAt:    make(map[int]float64, len(cutoffs)),
		NDCGAt:      make(map[int]float64, len(cutoffs)),
		HitRateAt:   make(map[int]float64, len(cutoffs)),
		PerQuery:    make([]RetrievalMetrics, 0, len(queries)),
	}
	for _, cutoff := range cutoffs {
		report.PrecisionAt[cutoff] = 0
		report.RecallAt[cutoff] = 0
		report.NDCGAt[cutoff] = 0
		report.HitRateAt[cutoff] = 0
	}
	judgmentsByQuery := make(map[string][]rag.Judgment)
	for _, judgment := range judgments {
		judgmentsByQuery[judgment.QueryID] = append(judgmentsByQuery[judgment.QueryID], judgment)
	}
	for _, query := range queries {
		queryJudgments := judgmentsByQuery[query.ID]
		ids, hasRanking := rankedIDs[query.ID]
		if len(queryJudgments) == 0 || !hasRanking {
			report.SkippedQueries++
			continue
		}
		metrics, err := Retrieval(query, ids, queryJudgments, cutoffs)
		if err != nil {
			return Report{}, err
		}
		report.PerQuery = append(report.PerQuery, metrics)
		report.MRR += metrics.MRR
		for _, cutoff := range cutoffs {
			report.PrecisionAt[cutoff] += metrics.PrecisionAt[cutoff]
			report.RecallAt[cutoff] += metrics.RecallAt[cutoff]
			report.NDCGAt[cutoff] += metrics.NDCGAt[cutoff]
			report.HitRateAt[cutoff] += metrics.HitRateAt[cutoff]
		}
	}
	report.EvaluatedQueries = len(report.PerQuery)
	if report.EvaluatedQueries == 0 {
		return report, nil
	}
	count := float64(report.EvaluatedQueries)
	report.MRR /= count
	for _, cutoff := range cutoffs {
		report.PrecisionAt[cutoff] /= count
		report.RecallAt[cutoff] /= count
		report.NDCGAt[cutoff] /= count
		report.HitRateAt[cutoff] /= count
	}
	return report, nil
}

// Retrieval evaluates ordered target IDs against judgments.
func Retrieval(query rag.Query, rankedTargetIDs []string, judgments []rag.Judgment, cutoffs []int) (RetrievalMetrics, error) {
	if err := validateCutoffs(cutoffs); err != nil {
		return RetrievalMetrics{}, err
	}
	relevant := map[string]float64{}
	target := ""
	for _, judgment := range judgments {
		if judgment.QueryID != query.ID {
			continue
		}
		if err := rag.Target(judgment.Target).Validate(); err != nil {
			return RetrievalMetrics{}, fmt.Errorf("query %q judgment target: %w", query.ID, err)
		}
		if judgment.TargetID == "" {
			return RetrievalMetrics{}, fmt.Errorf("query %q judgment target ID is required", query.ID)
		}
		if target == "" {
			target = judgment.Target
		} else if target != judgment.Target {
			return RetrievalMetrics{}, fmt.Errorf("query %q mixes relevance targets", query.ID)
		}
		if math.IsNaN(judgment.Grade) || math.IsInf(judgment.Grade, 0) || judgment.Grade < 0 {
			return RetrievalMetrics{}, fmt.Errorf("query %q target %q has invalid relevance grade %v", query.ID, judgment.TargetID, judgment.Grade)
		}
		if _, exists := relevant[judgment.TargetID]; exists {
			return RetrievalMetrics{}, fmt.Errorf("query %q has duplicate %s judgment for target %q", query.ID, judgment.Target, judgment.TargetID)
		}
		relevant[judgment.TargetID] = judgment.Grade
	}
	if len(relevant) == 0 {
		return RetrievalMetrics{}, fmt.Errorf("query %q has no relevance judgments", query.ID)
	}
	ids := unique(rankedTargetIDs)
	result := RetrievalMetrics{
		QueryID:     query.ID,
		Target:      target,
		PrecisionAt: map[int]float64{},
		RecallAt:    map[int]float64{},
		HitRateAt:   map[int]float64{},
		NDCGAt:      map[int]float64{},
		MRR:         reciprocalRank(ids, relevant),
	}
	for _, cutoff := range cutoffs {
		result.PrecisionAt[cutoff] = precision(ids, relevant, cutoff)
		result.RecallAt[cutoff] = recall(ids, relevant, cutoff)
		if result.RecallAt[cutoff] > 0 {
			result.HitRateAt[cutoff] = 1
		}
		result.NDCGAt[cutoff] = ndcg(ids, relevant, cutoff)
	}
	return result, nil
}

func validateCutoffs(cutoffs []int) error {
	seen := make(map[int]struct{}, len(cutoffs))
	for _, cutoff := range cutoffs {
		if cutoff < 1 {
			return fmt.Errorf("cutoffs must be positive")
		}
		if _, exists := seen[cutoff]; exists {
			return fmt.Errorf("duplicate retrieval cutoff %d", cutoff)
		}
		seen[cutoff] = struct{}{}
	}
	return nil
}

func unique(ids []string) []string {
	seen := map[string]bool{}
	result := make([]string, 0, len(ids))
	for _, id := range ids {
		if id != "" && !seen[id] {
			seen[id] = true
			result = append(result, id)
		}
	}
	return result
}

func precision(ids []string, relevant map[string]float64, cutoff int) float64 {
	limit := min(cutoff, len(ids))
	hits := 0
	for _, id := range ids[:limit] {
		if relevant[id] > 0 {
			hits++
		}
	}
	return float64(hits) / float64(cutoff)
}

func recall(ids []string, relevant map[string]float64, cutoff int) float64 {
	totalRelevant := 0
	for _, grade := range relevant {
		if grade > 0 {
			totalRelevant++
		}
	}
	if totalRelevant == 0 {
		return 0
	}
	hits := 0
	for _, id := range ids[:min(cutoff, len(ids))] {
		if relevant[id] > 0 {
			hits++
		}
	}
	return float64(hits) / float64(totalRelevant)
}

func reciprocalRank(ids []string, relevant map[string]float64) float64 {
	for index, id := range ids {
		if relevant[id] > 0 {
			return 1 / float64(index+1)
		}
	}
	return 0
}

func ndcg(ids []string, relevant map[string]float64, cutoff int) float64 {
	maxGrade := 0.0
	for _, grade := range relevant {
		maxGrade = max(maxGrade, grade)
	}
	var actual float64
	for index, id := range ids[:min(cutoff, len(ids))] {
		actual += gain(relevant[id], maxGrade, index)
	}
	grades := make([]float64, 0, len(relevant))
	for _, grade := range relevant {
		grades = append(grades, grade)
	}
	sort.Slice(grades, func(left, right int) bool { return grades[left] > grades[right] })
	var ideal float64
	for index, grade := range grades[:min(cutoff, len(grades))] {
		ideal += gain(grade, maxGrade, index)
	}
	if ideal == 0 {
		return 0
	}
	return actual / ideal
}

func gain(grade, scale float64, index int) float64 {
	// Scale every gain by the same 2^-scale factor. The factor cancels in
	// actual/ideal, while Exp2(grade-scale) cannot overflow. The Expm1 form
	// preserves a positive gain when grade is tiny: direct subtraction rounds
	// both terms to the same float64 value.
	numerator := math.Exp2(grade-scale) * -math.Expm1(-grade*math.Ln2)
	return numerator / math.Log2(float64(index+2))
}
