// Package flowpolicy contains ragkit-specific policy for Flowkit runs.
package flowpolicy

import (
	"context"
	"errors"
	"strings"

	"github.com/go-go-golems/flowkit/execution"
	"github.com/go-go-golems/flowkit/flow"
)

type transientMarker struct {
	marker   string
	incident string
}

// transientMarkers preserves the provider failure families learned from
// historical ragkit runs. Keep string matching here, next to the application
// that owns the policy, rather than in Flowkit's domain-neutral default.
var transientMarkers = []transientMarker{
	{"status=429", "provider concurrency/rate limit verdicts"},
	{"status=408", "provider request timeout verdicts"},
	{"status=5", "provider 5xx failures"},
	{"CONCURRENT_REQUEST_LIMIT", "gateway concurrency refusals during summary builds"},
	{"rate limit", "stringly rate-limit messages (lowercase)"},
	{"Rate limit", "stringly rate-limit messages (capitalized)"},
	{"connection reset", "transport drops mid-stream"},
	{"connection refused", "provider restarts between calls"},
	{"broken pipe", "transport drops mid-write"},
	{"unexpected EOF", "embeddings item 49 death after 13,847 completed summaries (2026-07-31)"},
	{"EOF", "dropped streams reported as bare EOF"},
	{"stream receive", "geppetto stream receive failures"},
	{"timeout", "transport timeouts (lowercase)"},
	{"Timeout", "transport timeouts (capitalized)"},
	{"timed out", "read connection timed out during judge execution (2026-07-31)"},
	{"network is unreachable", "local network/VPN blip during judge run (2026-07-31)"},
	{"no route to host", "local network/VPN blips"},
	{"stream error", "HTTP/2 stream resets from the peer"},
	{"INTERNAL_ERROR", "HTTP/2 INTERNAL_ERROR during screening (2026-07-31)"},
	{"temporarily unavailable", "provider-side temporary unavailability"},
	{"TLS handshake", "TLS handshake failures on flaky links"},
	{"tls: bad record MAC", "embedding TLS record failure (2026-07-31)"},
	{"generation response is empty", "all-reasoning generation stream (2026-07-31)"},
}

type classifier struct{}

// Classifier preserves ragkit's historical provider retry behavior while
// keeping Flowkit's default classifier domain-neutral.
var Classifier flow.Classifier = classifier{}

func (classifier) Classify(err error) flow.ErrorClass {
	if err == nil {
		return flow.Fatal
	}
	if flow.IsDataError(err) {
		return flow.DataError
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) ||
		errors.Is(err, execution.ErrBudgetExceeded) {
		return flow.Fatal
	}
	var status interface{ HTTPStatus() int }
	if errors.As(err, &status) {
		code := status.HTTPStatus()
		if code == 429 || code == 408 || (code >= 500 && code < 600) {
			return flow.Transient
		}
		return flow.Fatal
	}
	message := err.Error()
	if strings.Contains(message, "context canceled") || strings.Contains(message, "context deadline exceeded") {
		return flow.Fatal
	}
	for _, entry := range transientMarkers {
		if strings.Contains(message, entry.marker) {
			return flow.Transient
		}
	}
	return flow.Fatal
}

// Retry installs ragkit's classifier only when the caller did not choose one.
func Retry(spec flow.RetrySpec) flow.RetrySpec {
	if spec.Class == nil {
		spec.Class = Classifier
	}
	return spec
}

// Policy installs ragkit's classifier only when the caller did not choose one.
func Policy(policy flow.Policy) flow.Policy {
	policy.Retry = Retry(policy.Retry)
	return policy
}
