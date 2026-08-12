package indexbundle

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/stretchr/testify/require"
)

func TestStreamJSONArrayPreservesCanonicalSliceDigest(t *testing.T) {
	type item struct {
		ID   string `json:"id"`
		Text string `json:"text"`
	}
	want := []item{{ID: "a", Text: "first"}, {ID: "b", Text: "second"}}
	path := filepath.Join(t.TempDir(), "items.json")
	require.NoError(t, os.WriteFile(path, []byte("[\n {\"id\":\"a\",\"text\":\"first\"},\n {\"id\":\"b\",\"text\":\"second\"}\n]"), 0o600))

	var got []item
	count, gotDigest, err := streamJSONArray[item](t.Context(), path, func(_ int, value item) error {
		got = append(got, value)
		return nil
	})
	require.NoError(t, err)
	wantDigest, err := digest.JSON(want)
	require.NoError(t, err)
	require.Equal(t, len(want), count)
	require.Equal(t, want, got)
	require.Equal(t, wantDigest, gotDigest)
}

func TestStreamJSONArrayFailsClosed(t *testing.T) {
	type item struct {
		ID string `json:"id"`
	}
	for name, content := range map[string]string{
		"not-array":      `{"id":"a"}`,
		"unknown-field":  `[{"id":"a","extra":true}]`,
		"trailing-value": `[{"id":"a"}] {"extra":true}`,
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "items.json")
			require.NoError(t, os.WriteFile(path, []byte(content), 0o600))
			_, _, err := streamJSONArray[item](context.Background(), path, nil)
			require.Error(t, err)
		})
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	path := filepath.Join(t.TempDir(), "items.json")
	require.NoError(t, os.WriteFile(path, []byte(`[{"id":"a"}]`), 0o600))
	_, _, err := streamJSONArray[item](canceled, path, nil)
	require.ErrorIs(t, err, context.Canceled)
}
