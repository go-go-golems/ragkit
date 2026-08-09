package flow

import (
	"time"

	"github.com/go-go-golems/ragkit/execution"
)

// Policy is everything about HOW a step executes. Nothing in it may affect
// the step's cache identity.
type Policy struct {
	// Workers bounds concurrent Do calls. Values below one select one worker.
	Workers int
	// Admission declares the step's fail-closed resource plans. Each work
	// call (cache hits are free) consumes one unit of every declared
	// resource. Plans are validated before item one.
	Admission []Resource
	// Retry bounds the retry loop around Do.
	Retry RetrySpec
	// OnError chooses the destiny of a non-retryable item error.
	OnError FailureMode
}

// Resource is one named admission plan: the stated worst case, the admitted
// allowance, and optionally a unit price for the monetary preflight.
type Resource struct {
	Name    string
	Ceiling int
	Budget  int
	UnitUSD *float64
}

// plan converts to the execution layer's resource plan.
func (r Resource) plan() execution.ResourcePlan {
	return execution.ResourcePlan{
		Name:    r.Name,
		Ceiling: r.Ceiling,
		Budget:  r.Budget,
		UnitUSD: r.UnitUSD,
	}
}

// FailureMode distinguishes the three destinies of an item error that retry
// did not (or may not) absorb: kill the run, record the item as data, or
// drop it with a visible count.
type FailureMode int

const (
	// FailFast cancels the whole run on the first non-retryable item error.
	// The default: most steps (embeddings, indexing) have no meaningful
	// per-item degradation.
	FailFast FailureMode = iota
	// Quarantine turns an item error into a Quarantined record in the
	// results and report; the run continues and succeeds.
	Quarantine
	// Skip drops the item silently EXCEPT a count in the report and a
	// Skipped marker in its result slot. Rare; discouraged.
	Skip
)

// String names the mode for reports and logs.
func (mode FailureMode) String() string {
	switch mode {
	case FailFast:
		return "fail-fast"
	case Quarantine:
		return "quarantine"
	case Skip:
		return "skip"
	default:
		return "unknown"
	}
}

// RetrySpec bounds the retry loop around one item's Do.
type RetrySpec struct {
	// Attempts is the total number of tries, including the first. Zero or
	// one means no retry (today's generation.RetryOptions convention).
	Attempts int
	// Backoff shapes the delay between attempts.
	Backoff Backoff
	// Class classifies errors; nil selects DefaultClassifier.
	Class Classifier
}

// Backoff is exponential with jitter: the delay starts at Base, doubles per
// retry, gains up to 50% jitter, and never exceeds Cap. Zero values select
// the campaign defaults (base 2s, cap 1m).
type Backoff struct {
	Base time.Duration
	Cap  time.Duration
}

func (b Backoff) withDefaults() Backoff {
	if b.Base <= 0 {
		b.Base = 2 * time.Second
	}
	if b.Cap <= 0 {
		b.Cap = time.Minute
	}
	return b
}

// ItemError records one item's classified, non-retryable failure. Under
// Quarantine it is analysis data, not an error: the judge taxonomy showed
// that "bad item" is a result (DR-4).
type ItemError struct {
	Step     string     `json:"step"`
	Index    int        `json:"index"`
	Class    ErrorClass `json:"class"`
	Attempts int        `json:"attempts"`
	Message  string     `json:"message"`
}
