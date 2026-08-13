package indexbundle

import (
	"database/sql"
	"os"
	"path/filepath"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	"github.com/stretchr/testify/require"
)

func TestVerifyUsesProductionValidationAndReportsStableStages(t *testing.T) {
	input := fixtureInput(t, t.TempDir())
	built, err := Build(t.Context(), input)
	require.NoError(t, err)

	var stages []VerifyStage
	manifest, err := Verify(t.Context(), VerifyOptions{
		Path: built.Path, ExpectedBundleID: built.Manifest.BundleID,
		ExpectedCorpusPath: built.Manifest.CorpusPath,
		ScratchDirectory:   t.TempDir(),
		ObserveStage:       func(stage VerifyStage) { stages = append(stages, stage) },
	})
	require.NoError(t, err)
	require.Equal(t, built.Manifest.BundleID, manifest.BundleID)
	require.Equal(t, []VerifyStage{
		VerifyStageManifest,
		VerifyStageChunks,
		VerifyStageRepresentations,
		VerifyStageLexical,
		VerifyStageVector,
		VerifyStageComplete,
	}, stages)
}

func TestVerifyRejectsExpectedIdentityMismatch(t *testing.T) {
	input := fixtureInput(t, t.TempDir())
	built, err := Build(t.Context(), input)
	require.NoError(t, err)

	_, err = Verify(t.Context(), VerifyOptions{
		Path: built.Path, ExpectedBundleID: "rk-wrong", ScratchDirectory: t.TempDir(),
	})
	require.ErrorContains(t, err, "differs from expected")
}

func TestVerifyRejectsPersistedContentChanges(t *testing.T) {
	t.Run("payload rows", func(t *testing.T) {
		input := fixtureInput(t, t.TempDir())
		built, err := Build(t.Context(), input)
		require.NoError(t, err)

		db, err := sql.Open("sqlite3", filepath.Join(built.Path, contentName))
		require.NoError(t, err)
		_, err = db.Exec(`UPDATE document SET title = ? WHERE id = ?`, "Tampered", input.Documents[0].ID)
		require.NoError(t, err)
		require.NoError(t, db.Close())

		_, err = Verify(t.Context(), VerifyOptions{Path: built.Path, ScratchDirectory: t.TempDir()})
		require.ErrorContains(t, err, "content identity differs from manifest")

		_, err = Build(t.Context(), input)
		require.ErrorContains(t, err, "existing bundle identity is invalid")
		require.ErrorContains(t, err, "content identity differs from manifest")
	})

	t.Run("identity row", func(t *testing.T) {
		input := fixtureInput(t, t.TempDir())
		built, err := Build(t.Context(), input)
		require.NoError(t, err)

		db, err := sql.Open("sqlite3", filepath.Join(built.Path, contentName))
		require.NoError(t, err)
		_, err = db.Exec(`UPDATE content_identity SET content_digest = ? WHERE id = 1`, "tampered")
		require.NoError(t, err)
		require.NoError(t, db.Close())

		_, err = Verify(t.Context(), VerifyOptions{Path: built.Path, ScratchDirectory: t.TempDir()})
		require.ErrorContains(t, err, "content identity differs from manifest")
	})
}

func TestVerifyRequiresScratchDirectory(t *testing.T) {
	input := fixtureInput(t, t.TempDir())
	built, err := Build(t.Context(), input)
	require.NoError(t, err)

	_, err = Verify(t.Context(), VerifyOptions{Path: built.Path})
	require.ErrorContains(t, err, "scratch directory is required")
}

func TestVerifyUsesExplicitScratchOverTempDir(t *testing.T) {
	// Force Go's default temporary directory to an isolated location and prove
	// the verification relation is created under the explicit scratch
	// directory, not os.TempDir.
	t.Setenv("TMPDIR", t.TempDir())

	input := fixtureInput(t, t.TempDir())
	built, err := Build(t.Context(), input)
	require.NoError(t, err)

	scratch := t.TempDir()
	manifest, err := Verify(t.Context(), VerifyOptions{
		Path: built.Path, ExpectedBundleID: built.Manifest.BundleID,
		ScratchDirectory: scratch,
	})
	require.NoError(t, err)
	require.Equal(t, built.Manifest.BundleID, manifest.BundleID)

	// A successful Verify must leave no scratch relation files behind.
	entries, err := readDirNames(scratch)
	require.NoError(t, err)
	require.Empty(t, entries)
}

func readDirNames(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	return names, nil
}
