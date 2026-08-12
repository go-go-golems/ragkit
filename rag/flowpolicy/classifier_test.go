package flowpolicy

import (
	"context"
	"fmt"
	"testing"

	"github.com/go-go-golems/flowkit/flow"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/require"
)

func TestTransientMarkersPreserveHistoricalBehavior(t *testing.T) {
	seen := map[string]bool{}
	for _, entry := range transientMarkers {
		require.NotEmpty(t, entry.marker)
		require.NotEmpty(t, entry.incident)
		require.False(t, seen[entry.marker], "duplicate marker %q", entry.marker)
		seen[entry.marker] = true
		err := fmt.Errorf("provider failed: %s (wrapped)", entry.marker)
		require.Equal(t, flow.Transient, Classifier.Classify(err), entry.incident)
	}
}

func TestClassifierPreservesTypedAndCancellationPrecedence(t *testing.T) {
	require.Equal(t, flow.Transient, Classifier.Classify(&flow.StatusError{Status: 503, Err: errors.New("down")}))
	require.Equal(t, flow.Fatal, Classifier.Classify(&flow.StatusError{Status: 401, Err: errors.New("bad key")}))
	require.Equal(t, flow.Fatal, Classifier.Classify(errors.Wrap(context.Canceled, "timeout")))
	require.Equal(t, flow.DataError, Classifier.Classify(flow.AsDataError(errors.New("unexpected EOF"))))
	require.Equal(t, flow.Fatal, Classifier.Classify(errors.New("unknown")))
}

func TestPolicyDoesNotOverrideCallerClassifier(t *testing.T) {
	custom := flow.ClassifierFunc(func(error) flow.ErrorClass { return flow.DataError })
	policy := Policy(flow.Policy{Retry: flow.RetrySpec{Class: custom}})
	require.Equal(t, flow.DataError, policy.Retry.Class.Classify(errors.New("unexpected EOF")))
}
