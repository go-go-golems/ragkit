package flow

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/pkg/errors"
	"github.com/stretchr/testify/require"

	"github.com/go-go-golems/ragkit/execution"
)

func TestErrorClassJSONRoundTrip(t *testing.T) {
	for _, class := range []ErrorClass{Transient, DataError, Fatal} {
		encoded, err := json.Marshal(class)
		require.NoError(t, err)
		var decoded ErrorClass
		require.NoError(t, json.Unmarshal(encoded, &decoded))
		require.Equal(t, class, decoded)
	}

	var decoded ErrorClass
	require.Error(t, json.Unmarshal([]byte(`"unknown"`), &decoded))
}

// Every fallback marker must name the incident that earned its place: the
// table grew one corpse at a time, and this test keeps that discipline.
func TestTransientMarkersEachNameTheirIncident(t *testing.T) {
	require.NotEmpty(t, transientMarkers)
	seen := map[string]bool{}
	for _, entry := range transientMarkers {
		require.NotEmpty(t, entry.marker, "marker text is required")
		require.NotEmpty(t, entry.incident, "marker %q must name its incident", entry.marker)
		require.False(t, seen[entry.marker], "marker %q is duplicated", entry.marker)
		seen[entry.marker] = true
	}
}

func TestDefaultClassifierSubstringTable(t *testing.T) {
	for _, entry := range transientMarkers {
		err := fmt.Errorf("provider call failed: %s (wrapped)", entry.marker)
		require.Equal(t, Transient, DefaultClassifier.Classify(err),
			"marker %q (incident: %s)", entry.marker, entry.incident)
	}
}

func TestDefaultClassifierTypedStatuses(t *testing.T) {
	cases := []struct {
		status int
		class  ErrorClass
	}{
		{429, Transient},
		{408, Transient},
		{500, Transient},
		{503, Transient},
		{400, Fatal},
		{404, Fatal},
		{401, Fatal},
	}
	for _, testCase := range cases {
		err := errors.Wrap(&StatusError{Status: testCase.status, Err: errors.New("nope")}, "call")
		require.Equal(t, testCase.class, DefaultClassifier.Classify(err), "status %d", testCase.status)
	}
}

func TestDefaultClassifierNeverRetriesCancellation(t *testing.T) {
	require.Equal(t, Fatal, DefaultClassifier.Classify(context.Canceled))
	require.Equal(t, Fatal, DefaultClassifier.Classify(context.DeadlineExceeded))
	require.Equal(t, Fatal, DefaultClassifier.Classify(errors.Wrap(context.Canceled, "generate")))
	// Stringly cancellation from partly-stringly provider errors: even when
	// the message also contains a transient marker ("timeout"), the caller
	// verdict wins.
	require.Equal(t, Fatal, DefaultClassifier.Classify(errors.New("stream: context deadline exceeded (Client.Timeout)")))
}

func TestDefaultClassifierBudgetExhaustionIsFatal(t *testing.T) {
	err := errors.Wrap(execution.ErrBudgetExceeded, "wait for resources")
	require.Equal(t, Fatal, DefaultClassifier.Classify(err))
}

func TestDefaultClassifierDataErrorMarker(t *testing.T) {
	parseErr := AsDataError(errors.New("judge verdict JSON missing 'verdicts' array"))
	require.Equal(t, DataError, DefaultClassifier.Classify(parseErr))
	require.Equal(t, DataError, DefaultClassifier.Classify(errors.Wrap(parseErr, "statements pass")))
	require.True(t, IsDataError(parseErr))
	require.Nil(t, AsDataError(nil))
	// The marker must beat the substring table: a malformed response whose
	// message mentions "EOF" is still a data verdict, not a transient.
	require.Equal(t, DataError, DefaultClassifier.Classify(AsDataError(errors.New("unexpected EOF while parsing"))))
}

func TestDefaultClassifierUnknownErrorsAreFatal(t *testing.T) {
	require.Equal(t, Fatal, DefaultClassifier.Classify(errors.New("something entirely new")))
}
