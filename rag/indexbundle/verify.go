package indexbundle

import (
	"context"
	"strings"

	"github.com/pkg/errors"
)

// Verify validates an immutable bundle without opening its serving indexes or
// requiring an embedding provider. It is read-only and fail closed.
func Verify(ctx context.Context, options VerifyOptions) (Manifest, error) {
	if err := ctx.Err(); err != nil {
		return Manifest{}, err
	}
	if strings.TrimSpace(options.Path) == "" {
		return Manifest{}, errors.New("bundle verification path is required")
	}
	state, err := loadVerifiedManifest(ctx, options.Path)
	if err != nil {
		return Manifest{}, err
	}
	manifest := state.manifest
	if expected := strings.TrimSpace(options.ExpectedBundleID); expected != "" && manifest.BundleID != expected {
		return Manifest{}, errors.Errorf("bundle ID %q differs from expected %q", manifest.BundleID, expected)
	}
	if expected := strings.TrimSpace(options.ExpectedCorpusPath); expected != "" && manifest.CorpusPath != expected {
		return Manifest{}, errors.Errorf("bundle corpus path %q differs from expected %q", manifest.CorpusPath, expected)
	}
	observeVerifyStage(options, VerifyStageManifest)

	relation, err := openVerificationRelation(ctx)
	if err != nil {
		return Manifest{}, err
	}
	defer func() { _ = relation.closeAndRemove() }()

	chunks, err := streamVerifiedChunks(ctx, state, relation)
	if err != nil {
		return Manifest{}, err
	}
	observeVerifyStage(options, VerifyStageChunks)

	verified, err := streamVerifiedStoredIdentity(ctx, chunks)
	if err != nil {
		return Manifest{}, err
	}
	observeVerifyStage(options, VerifyStageRepresentations)
	if err := relation.closeAndRemove(); err != nil {
		return Manifest{}, err
	}

	if err := validateContentBackendIdentity(ctx, verified); err != nil {
		return Manifest{}, err
	}
	if err := validateLexicalBackendIdentity(ctx, verified); err != nil {
		return Manifest{}, err
	}
	observeVerifyStage(options, VerifyStageLexical)
	if err := validateVectorBackendIdentity(ctx, verified); err != nil {
		return Manifest{}, err
	}
	if manifest.Vector != nil {
		observeVerifyStage(options, VerifyStageVector)
	}
	observeVerifyStage(options, VerifyStageComplete)
	return manifest, nil
}

func observeVerifyStage(options VerifyOptions, stage VerifyStage) {
	if options.ObserveStage != nil {
		options.ObserveStage(stage)
	}
}
