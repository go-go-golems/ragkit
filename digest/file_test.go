package digest

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFileMatchesBytesAndHonorsCancellation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "input")
	content := []byte("digest me")
	require.NoError(t, os.WriteFile(path, content, 0o600))
	value, err := File(t.Context(), path)
	require.NoError(t, err)
	require.Equal(t, Bytes(content), value)

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = File(ctx, path)
	require.ErrorIs(t, err, context.Canceled)
}
