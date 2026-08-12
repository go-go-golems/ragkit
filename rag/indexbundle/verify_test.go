package indexbundle

import (
	"database/sql"
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

	_, err = Verify(t.Context(), VerifyOptions{Path: built.Path, ExpectedBundleID: "rk-wrong"})
	require.ErrorContains(t, err, "differs from expected")
}

func TestVerifyRejectsPersistedContentChanges(t *testing.T) {
	input := fixtureInput(t, t.TempDir())
	built, err := Build(t.Context(), input)
	require.NoError(t, err)

	db, err := sql.Open("sqlite3", filepath.Join(built.Path, contentName))
	require.NoError(t, err)
	_, err = db.Exec(`UPDATE document SET title = ? WHERE id = ?`, "Tampered", input.Documents[0].ID)
	require.NoError(t, err)
	require.NoError(t, db.Close())

	_, err = Verify(t.Context(), VerifyOptions{Path: built.Path})
	require.ErrorContains(t, err, "content identity differs from manifest")

	_, err = Build(t.Context(), input)
	require.ErrorContains(t, err, "existing bundle identity is invalid")
	require.ErrorContains(t, err, "content identity differs from manifest")
}
