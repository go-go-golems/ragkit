package indexbundle

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPreflightScratchSucceedsAndLeavesNoFiles(t *testing.T) {
	scratch := t.TempDir()
	result, err := PreflightScratch(t.Context(), scratch)
	require.NoError(t, err)
	require.Equal(t, scratch, result.Directory)
	require.Positive(t, result.Elapsed)

	entries, err := os.ReadDir(scratch)
	require.NoError(t, err)
	require.Empty(t, entries)
}

func TestPreflightScratchCreatesMissingDirectory(t *testing.T) {
	scratch := filepath.Join(t.TempDir(), "nested", "scratch")
	result, err := PreflightScratch(t.Context(), scratch)
	require.NoError(t, err)
	require.Equal(t, scratch, result.Directory)

	info, err := os.Stat(scratch)
	require.NoError(t, err)
	require.True(t, info.IsDir())
}

func TestPreflightScratchRequiresNonEmptyDirectory(t *testing.T) {
	_, err := PreflightScratch(t.Context(), "  ")
	require.ErrorContains(t, err, "scratch directory is required")
}

func TestPreflightScratchRejectsUnwritableDirectory(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("permission test is not meaningful when running as root")
	}
	parent := t.TempDir()
	scratch := filepath.Join(parent, "scratch")
	require.NoError(t, os.MkdirAll(scratch, 0o500)) // read+execute, no write
	_, err := PreflightScratch(t.Context(), scratch)
	require.Error(t, err)
}
