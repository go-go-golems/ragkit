package flow

import (
	"context"

	"github.com/go-go-golems/ragkit/execution"
)

// Meters aggregates named usage counters (tokens, cost) generically; domain
// adapters translate their provider usage into meter names. Only fresh work
// is metered — cache hits are free and stay free in the accounting.
type Meters map[string]float64

// Add merges other into meters, summing shared names.
func (meters Meters) Add(other Meters) {
	for name, value := range other {
		meters[name] += value
	}
}

// StepReport counts one step's execution: items seen, cache traffic, retry
// pressure, quarantine and skip decisions, admission spend, and meters.
type StepReport struct {
	Items          int                                 `json:"items"`
	Hits           int                                 `json:"hits"`
	Misses         int                                 `json:"misses"`
	Stored         int                                 `json:"stored"`
	WorkCalls      int                                 `json:"work_calls"`
	Retries        int                                 `json:"retries"`
	Quarantined    int                                 `json:"quarantined"`
	Skipped        int                                 `json:"skipped"`
	RetriesByClass map[string]int                      `json:"retries_by_class,omitempty"`
	Spend          map[string]execution.BudgetSnapshot `json:"spend,omitempty"`
	Meters         Meters                              `json:"meters,omitempty"`
}

// merge adds other's counts into the report (same step name run twice).
func (report *StepReport) merge(other StepReport) {
	report.Items += other.Items
	report.Hits += other.Hits
	report.Misses += other.Misses
	report.Stored += other.Stored
	report.WorkCalls += other.WorkCalls
	report.Retries += other.Retries
	report.Quarantined += other.Quarantined
	report.Skipped += other.Skipped
	for class, count := range other.RetriesByClass {
		if report.RetriesByClass == nil {
			report.RetriesByClass = map[string]int{}
		}
		report.RetriesByClass[class] += count
	}
	for name, snapshot := range other.Spend {
		if report.Spend == nil {
			report.Spend = map[string]execution.BudgetSnapshot{}
		}
		report.Spend[name] = snapshot
	}
	if len(other.Meters) > 0 {
		if report.Meters == nil {
			report.Meters = Meters{}
		}
		report.Meters.Add(other.Meters)
	}
}

// Report is the unified execution report: one StepReport per step name.
type Report struct {
	Steps map[string]StepReport `json:"steps"`
}

// merge folds other's steps into the report.
func (report *Report) merge(other Report) {
	for name, step := range other.Steps {
		if report.Steps == nil {
			report.Steps = map[string]StepReport{}
		}
		merged := report.Steps[name]
		merged.merge(step)
		report.Steps[name] = merged
	}
}

// Step returns the named step's report (zero value when absent).
func (report Report) Step(name string) StepReport {
	return report.Steps[name]
}

// Result carries one item's outcome, position-aligned with the input:
// Results[i] always corresponds to items[i] — evaluation joins depend on it.
type Result[O any] struct {
	Value O `json:"value"`
	// Cache records the durable state of this item (hit/stored/pending).
	// Zero for uncached steps.
	Cache execution.CacheOutcome `json:"cache,omitempty"`
	// Quarantined is set iff the step quarantined this item; Value is the
	// zero value then.
	Quarantined *ItemError `json:"quarantined,omitempty"`
	// Skipped marks an item dropped by FailureMode Skip.
	Skipped bool `json:"skipped,omitempty"`
}

// EventType names one ledger event.
type EventType string

const (
	// EventHit records a cache hit (free).
	EventHit EventType = "hit"
	// EventStored records fresh work committed to the store.
	EventStored EventType = "stored"
	// EventDone records fresh uncached work.
	EventDone EventType = "done"
	// EventRetry records one retry attempt about to back off.
	EventRetry EventType = "retry"
	// EventQuarantined records an item error kept as data.
	EventQuarantined EventType = "quarantined"
	// EventSkipped records an item dropped by policy.
	EventSkipped EventType = "skipped"
)

// Event is one observable moment of a run, suitable for appending to an
// experiment run's JSONL journal.
type Event struct {
	Step    string    `json:"step"`
	Index   int       `json:"index"`
	Type    EventType `json:"type"`
	Class   string    `json:"class,omitempty"`
	Attempt int       `json:"attempt,omitempty"`
	Error   string    `json:"error,omitempty"`
}

// Ledger consumes run events. Implementations typically adapt
// experiment.Run.AppendJSONL; a ledger error fails the run (journals are
// part of the record, not best-effort).
type Ledger interface {
	Event(ctx context.Context, event Event) error
}
