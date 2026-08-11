package indexbundle

import (
	"testing"

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
