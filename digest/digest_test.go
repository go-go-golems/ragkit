package digest

import (
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
