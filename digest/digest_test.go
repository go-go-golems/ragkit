package digest

import (
	"context"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBytesAndTextGolden(t *testing.T) {
	const expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
	require.Equal(t, expected, Bytes([]byte("hello")))
	require.Equal(t, expected, Text("hello"))
	require.Equal(t, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Text(""))
	require.Equal(t, "5af3c1a8aa660fab6c1ff1c61febd348ab3529e86fec4762467f682339bef201", Text("🌳"))
}

func TestJSONSequenceMatchesSliceDigest(t *testing.T) {
	values := []map[string]any{
		{"id": "a", "value": 1},
		{"id": "b", "value": "two"},
	}
	want, err := JSON(values)
	require.NoError(t, err)
	got, err := JSONSequence(t.Context(), func(yield func(map[string]any) error) error {
		for _, value := range values {
			if err := yield(value); err != nil {
				return err
			}
		}
		return nil
	})
	require.NoError(t, err)
	require.Equal(t, want, got)

	empty, err := JSONSequence[map[string]any](t.Context(), func(func(map[string]any) error) error { return nil })
	require.NoError(t, err)
	wantEmpty, err := JSON([]map[string]any{})
	require.NoError(t, err)
	require.Equal(t, wantEmpty, empty)
}

func TestJSONSequenceFailsClosed(t *testing.T) {
	_, err := JSONSequence[int](t.Context(), nil)
	require.ErrorContains(t, err, "producer is required")

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = JSONSequence(ctx, func(yield func(int) error) error { return yield(1) })
	require.ErrorIs(t, err, context.Canceled)

	_, err = JSONSequence(t.Context(), func(yield func(float64) error) error {
		return yield(math.Inf(1))
	})
	require.ErrorContains(t, err, "marshal JSON digest sequence value")
}

func TestJSONIsDeterministicAcrossMapInsertionOrder(t *testing.T) {
	left := map[string]int{"b": 2, "a": 1}
	right := map[string]int{"a": 1, "b": 2}
	leftDigest, err := JSON(left)
	require.NoError(t, err)
	rightDigest, err := JSON(right)
	require.NoError(t, err)
	require.Equal(t, leftDigest, rightDigest)
	require.Equal(t, "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777", leftDigest)
}

func TestJSONReportsMarshalFailure(t *testing.T) {
	_, err := JSON(math.Inf(1))
	require.ErrorContains(t, err, "marshal digest input")
}

func TestTruncatedJSON(t *testing.T) {
	full, err := JSON(map[string]string{"id": "review"})
	require.NoError(t, err)
	truncated, err := TruncatedJSON("review-", 12, map[string]string{"id": "review"})
	require.NoError(t, err)
	require.Equal(t, "review-"+full[:24], truncated)

	_, err = TruncatedJSON("", 0, nil)
	require.Error(t, err)
	_, err = TruncatedJSON("", 33, nil)
	require.Error(t, err)
}
