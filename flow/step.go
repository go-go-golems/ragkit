package flow

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/go-go-golems/flowkit/execution"
	"github.com/go-go-golems/ragkit/digest"
)

// Identity names the durable cache identity of a step. Kind and Version map
// onto the execution cache key's Step and Version fields, so every step's
// keyspace is disjoint by construction, and Key returns the exact bytes the
// result depends on — the content the model sees, byte-exact. Nothing about
// execution (workers, retry, reasoning settings) may leak into it: that is
// the discipline that keeps replays digit-exact and kills cheap.
//
// A step with Kind == "" is uncached (pure compute, e.g. parsing).
type Identity[I any] struct {
	Kind    string
	Version string
	Key     func(I) ([]byte, error)
}

// Step is one typed unit of expensive work: its identity (cache key), its
// policy (workers, admission, retry, failure mode), and its function. Steps
// are values; composition (Pipe, Batched) returns new Steps.
type Step[I, O any] struct {
	// Name is the step's report and log name, and by convention the prefix
	// of its admission resource names ("judge-statements").
	Name     string
	Identity Identity[I]
	Policy   Policy

	// Barrier makes a pipeline stage wait for every upstream result before
	// processing its first item. Needed only when the stage consumes
	// cross-item state; per-item streaming is the default.
	Barrier bool

	// Do performs the work for one item.
	Do func(context.Context, I) (O, error)

	// Meter optionally converts one fresh (non-cached) result into named
	// counters aggregated in the step's report. Cache hits are not metered:
	// meters record what this run actually spent.
	Meter func(O) Meters

	// AttemptMeter optionally meters every fresh Do result, including results
	// returned with an error. Use it when a provider reports billable usage on
	// failed attempts. A step should set either Meter or AttemptMeter, not both.
	AttemptMeter func(O, error) Meters

	// OnResult optionally observes every successful item (cache hits and
	// fresh work, never quarantined or skipped items) as it completes, in
	// completion order, with its input index and cache outcome. Streaming
	// artifact writers hang here; an error fails the run.
	OnResult func(context.Context, int, O, execution.CacheOutcome) error

	// override replaces the standard per-item engine (set by Batched).
	override func(context.Context, []I, Options, func(context.Context, int, O, execution.CacheOutcome) error) ([]Result[O], Report, error)
	// extraPlans declares admission resources consumed by an override's
	// nested steps (Batched repairs) so the shared preflight sees them
	// before item one.
	extraPlans []stagePlan
	// extraPolicies names nested policies consumed by an override so they are
	// validated before the override performs any provider work.
	extraPolicies []policySpec
	// stages is the flattened pipeline representation (set by Pipe*).
	stages []stageSpec
}

// Key returns the execution cache key for one item of a cached step. The
// mapping is direct — Kind and Version become the key's Step and Version and
// the key bytes are digested unmodified — so a step whose Identity mirrors an
// existing cache family (generation, embeddings) produces byte-identical keys
// and replays its historical entries (DR-3).
func (s Step[I, O]) Key(item I) (execution.Key, error) {
	if s.Identity.Kind == "" {
		return execution.Key{}, fmt.Errorf("step %q has no cache identity", s.Name)
	}
	if strings.TrimSpace(s.Identity.Version) == "" {
		return execution.Key{}, fmt.Errorf("step %q identity version is required", s.Name)
	}
	if s.Identity.Key == nil {
		return execution.Key{}, fmt.Errorf("step %q identity key function is required", s.Name)
	}
	keyBytes, err := s.Identity.Key(item)
	if err != nil {
		return execution.Key{}, fmt.Errorf("step %q: build identity key: %w", s.Name, err)
	}
	return execution.Key{
		Step:        s.Identity.Kind,
		Version:     s.Identity.Version,
		InputDigest: digest.Bytes(keyBytes),
	}, nil
}

// keyDigest reproduces execution's key file digest (digest of the marshaled
// key) for cache outcomes and in-flight deduplication.
func keyDigest(key execution.Key) (string, error) {
	data, err := json.Marshal(key)
	if err != nil {
		return "", fmt.Errorf("marshal cache key: %w", err)
	}
	return digest.Bytes(data), nil
}
