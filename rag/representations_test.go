package rag

import (
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/stretchr/testify/require"
)

func TestRawRepresentationsAreStableAndSensitive(t *testing.T) {
	chunks := []Chunk{{ID: "chunk", Text: "text", ContentDigest: digest.Text("text")}}
	first, err := RawRepresentations(chunks)
	require.NoError(t, err)
	second, err := RawRepresentations(chunks)
	require.NoError(t, err)
	require.Equal(t, first, second)
	require.Equal(t, "raw", first[0].Kind)
	require.Equal(t, "rep-3fa424a352378dbc", first[0].ID)
	require.Equal(t, "982d9e3eb996f559e633f4d194def3761d909f5a3b647d1a851fead67c32c9d1", first[0].ContentDigest)
	chunks[0].ContentDigest = digest.Text("changed")
	changed, err := RawRepresentations(chunks)
	require.NoError(t, err)
	require.NotEqual(t, first[0].ID, changed[0].ID)
}
