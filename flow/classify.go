package flow

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/go-go-golems/ragkit/execution"
)

// ErrorClass is the destiny of one item error.
type ErrorClass int

const (
	// Transient errors retry with backoff.
	Transient ErrorClass = iota
	// DataError is a well-transported but malformed response: never retried,
	// honored per the step's FailureMode. Providers cannot produce it — the
	// step's parser does, by wrapping with AsDataError.
	DataError
	// Fatal errors always fail the run, regardless of FailureMode.
	Fatal
)

// String names the class for reports and logs.
func (class ErrorClass) String() string {
	switch class {
	case Transient:
		return "transient"
	case DataError:
		return "data"
	case Fatal:
		return "fatal"
	default:
		return "unknown"
	}
}

// MarshalText serializes the class name into quarantine records.
func (class ErrorClass) MarshalText() ([]byte, error) {
	if class < Transient || class > Fatal {
		return nil, fmt.Errorf("unknown error class %d", class)
	}
	return []byte(class.String()), nil
}

// UnmarshalText restores the stable names emitted by MarshalText.
func (class *ErrorClass) UnmarshalText(text []byte) error {
	if class == nil {
		return errors.New("error class destination is nil")
	}
	switch string(text) {
	case "transient":
		*class = Transient
	case "data":
		*class = DataError
	case "fatal":
		*class = Fatal
	default:
		return fmt.Errorf("unknown error class %q", text)
	}
	return nil
}

// Classifier assigns one item error its class.
type Classifier interface {
	Classify(error) ErrorClass
}

// ClassifierFunc adapts a function to Classifier.
type ClassifierFunc func(error) ErrorClass

// Classify implements Classifier.
func (f ClassifierFunc) Classify(err error) ErrorClass { return f(err) }

// dataError marks a parser verdict on a well-transported response.
type dataError struct{ err error }

func (e *dataError) Error() string { return e.err.Error() }
func (e *dataError) Unwrap() error { return e.err }

// AsDataError wraps a step parser's error so the classifier files it as
// DataError instead of Fatal. A nil error stays nil.
func AsDataError(err error) error {
	if err == nil {
		return nil
	}
	return &dataError{err: err}
}

// IsDataError reports whether err carries the DataError marker.
func IsDataError(err error) bool {
	var marker *dataError
	return errors.As(err, &marker)
}

// StatusError is a typed provider error carrying an HTTP status. Adapters
// that know their transport status should wrap with it so classification
// stops depending on substrings (tier one of DefaultClassifier).
type StatusError struct {
	Status int
	Err    error
}

// Error implements error.
func (e *StatusError) Error() string {
	return fmt.Sprintf("status=%d: %v", e.Status, e.Err)
}

// Unwrap exposes the wrapped error.
func (e *StatusError) Unwrap() error { return e.Err }

// HTTPStatus implements the typed-status interface tier one matches on.
func (e *StatusError) HTTPStatus() int { return e.Status }

// transientMarker is one entry of the single shared substring fallback
// table. Every entry names the incident (or failure family) that earned its
// place; the table's test enforces that. Adding a marker anywhere else in
// the repository is a review smell.
type transientMarker struct {
	marker   string
	incident string
}

// transientMarkers is migrated verbatim from generation/retry.go (the
// pre-flow retry wrapper), incident comments preserved as incident names.
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
	{"timed out", `"read: connection timed out" — killed judge relaunch at item 3 (2026-07-31)`},
	{"network is unreachable", "local network/VPN blip during judge run (2026-07-31)"},
	{"no route to host", "local network/VPN blips"},
	{"stream error", "HTTP/2 stream resets from the peer"},
	{"INTERNAL_ERROR", `HTTP/2 "INTERNAL_ERROR" that killed screening at llm-chunk (2026-07-31)`},
	{"temporarily unavailable", "provider-side temporary unavailability"},
	{"TLS handshake", "TLS handshake failures on flaky links"},
	// OpenAI embeddings over a flaky link can fail mid-record, not just at
	// handshake; Go reports it as "remote error: tls: bad record MAC".
	{"tls: bad record MAC", "corpus-2000 build died at embeddings item 81 on a TLS record error (2026-07-31, tmux-logs/build-2000-final.log)"},
	// Reasoning-mode models occasionally emit a stream that is all hidden
	// reasoning and no content; the adapter reports it as an empty response.
	// A deterministic empty burns the attempts and still fails loudly.
	{"generation response is empty", "judge run 3566e8654e92 died at item 308 on an all-reasoning stream (2026-07-31)"},
}

// defaultClassifier implements the three-tier classification described in
// the RAG-TTC-FLOW-001 design doc §5.3.
type defaultClassifier struct{}

// DefaultClassifier is the shared classifier: (1) typed statuses via
// errors.As — 429/408/5xx transient, other statuses fatal; (2) context
// cancellation and budget exhaustion are fatal, never retried; (3) the ONE
// substring fallback table, migrated verbatim from generation/retry.go.
// Anything unrecognized is fatal: fail closed, visibly.
var DefaultClassifier Classifier = defaultClassifier{}

// Classify implements Classifier.
func (defaultClassifier) Classify(err error) ErrorClass {
	if err == nil {
		return Fatal
	}
	if IsDataError(err) {
		return DataError
	}
	// Tier two first for typed sentinels: cancellation is a caller verdict
	// and budget exhaustion is the experiment ceiling — retrying either
	// would fight the operator.
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return Fatal
	}
	if errors.Is(err, execution.ErrBudgetExceeded) {
		return Fatal
	}
	// Tier one: typed provider statuses.
	var status interface{ HTTPStatus() int }
	if errors.As(err, &status) {
		code := status.HTTPStatus()
		switch {
		case code == 429 || code == 408 || code >= 500:
			return Transient
		default:
			return Fatal
		}
	}
	// Stringly cancellation (geppetto errors are partly stringly and can
	// bury the typed chain).
	message := err.Error()
	if strings.Contains(message, "context canceled") ||
		strings.Contains(message, "context deadline exceeded") {
		return Fatal
	}
	// Tier three: the single substring fallback table.
	for _, entry := range transientMarkers {
		if strings.Contains(message, entry.marker) {
			return Transient
		}
	}
	return Fatal
}
