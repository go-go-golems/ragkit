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
	report := Report{
		RecallAt:  make(map[int]float64, len(cutoffs)),
		NDCGAt:    make(map[int]float64, len(cutoffs)),
		HitRateAt: make(map[int]float64, len(cutoffs)),
		PerQuery:  make([]RetrievalMetrics, 0, len(queries)),
	}
	for _, cutoff := range cutoffs {
		if cutoff < 1 {
			return Report{}, fmt.Errorf("cutoffs must be positive")
		}
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
		report.RecallAt[cutoff] /= count
		report.NDCGAt[cutoff] /= count
		report.HitRateAt[cutoff] /= count
	}
	return report, nil
}

// Retrieval evaluates ordered target IDs against judgments.
func Retrieval(query rag.Query, rankedTargetIDs []string, judgments []rag.Judgment, cutoffs []int) (RetrievalMetrics, error) {
	relevant := map[string]float64{}
	target := ""
	for _, judgment := range judgments {
		if judgment.QueryID != query.ID {
			continue
		}
		if target == "" {
			target = judgment.Target
		} else if target != judgment.Target {
			return RetrievalMetrics{}, fmt.Errorf("query %q mixes relevance targets", query.ID)
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
		if cutoff < 1 {
			return RetrievalMetrics{}, fmt.Errorf("cutoffs must be positive")
		}
		result.PrecisionAt[cutoff] = precision(ids, relevant, cutoff)
		result.RecallAt[cutoff] = recall(ids, relevant, cutoff)
		if result.RecallAt[cutoff] > 0 {
			result.HitRateAt[cutoff] = 1
		}
		result.NDCGAt[cutoff] = ndcg(ids, relevant, cutoff)
	}
	return result, nil
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
	var actual float64
	for index, id := range ids[:min(cutoff, len(ids))] {
		actual += gain(relevant[id], index)
	}
	grades := make([]float64, 0, len(relevant))
	for _, grade := range relevant {
		grades = append(grades, grade)
	}
	sort.Slice(grades, func(left, right int) bool { return grades[left] > grades[right] })
	var ideal float64
	for index, grade := range grades[:min(cutoff, len(grades))] {
		ideal += gain(grade, index)
	}
	if ideal == 0 {
		return 0
	}
	return actual / ideal
}

func gain(grade float64, index int) float64 {
	return (math.Pow(2, grade) - 1) / math.Log2(float64(index+2))
}
