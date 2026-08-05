package fsutil

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAtomicWritePublishesAndReplaces(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "nested", "value")
	require.NoError(t, AtomicWrite(t.Context(), path, []byte("first"), AtomicWriteOptions{}))
	require.NoError(t, AtomicWrite(t.Context(), path, []byte("second"), AtomicWriteOptions{}))
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, "second", string(data))
	info, err := os.Stat(path)
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0o600), info.Mode().Perm())
	parent, err := os.Stat(filepath.Dir(path))
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0o700), parent.Mode().Perm())
}

func TestAtomicWriteCancellationAndCleanup(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	root := t.TempDir()
	require.ErrorIs(t, AtomicWrite(ctx, filepath.Join(root, "value"), nil, AtomicWriteOptions{}), context.Canceled)

	destination := filepath.Join(root, "destination")
	require.NoError(t, os.Mkdir(destination, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(destination, "child"), nil, 0o600))
	err := AtomicWrite(t.Context(), destination, []byte("value"), AtomicWriteOptions{TempPattern: ".test-*"})
	require.Error(t, err)
	matches, globErr := filepath.Glob(filepath.Join(root, ".test-*"))
	require.NoError(t, globErr)
	require.Empty(t, matches)
}

func TestJoinWithin(t *testing.T) {
	t.Parallel()
	root := filepath.Join("tmp", "root")
	path, err := JoinWithin(root, "nested/value")
	require.NoError(t, err)
	require.Equal(t, filepath.Join(root, "nested", "value"), path)
	normalized, err := JoinWithin(filepath.Join(root, "."), "value")
	require.NoError(t, err)
	require.Equal(t, filepath.Join(root, "value"), normalized)
	for _, invalid := range []string{"", ".", "..", "../value", filepath.Join("..", "nested", "value")} {
		_, err := JoinWithin(root, invalid)
		require.Error(t, err)
	}
	_, err = JoinWithin(root, filepath.Join(string(filepath.Separator), "absolute"))
	require.Error(t, err)
}

func TestDirectorySize(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	size, err := DirectorySize(t.Context(), root)
	require.NoError(t, err)
	require.Zero(t, size)
	require.NoError(t, os.Mkdir(filepath.Join(root, "nested"), 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(root, "a"), []byte("123"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(root, "nested", "b"), []byte("4567"), 0o600))
	size, err = DirectorySize(t.Context(), root)
	require.NoError(t, err)
	require.Equal(t, int64(7), size)
	if err := os.Symlink(filepath.Join(root, "nested", "b"), filepath.Join(root, "link")); err == nil {
		info, statErr := os.Lstat(filepath.Join(root, "link"))
		require.NoError(t, statErr)
		size, err = DirectorySize(t.Context(), root)
		require.NoError(t, err)
		require.Equal(t, int64(7)+info.Size(), size)
	}
	_, err = DirectorySize(t.Context(), filepath.Join(root, "missing"))
	require.True(t, errors.Is(err, os.ErrNotExist))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = DirectorySize(ctx, root)
	require.ErrorIs(t, err, context.Canceled)
}
