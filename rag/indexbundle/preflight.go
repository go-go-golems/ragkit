package indexbundle

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/pkg/errors"
)

// ScratchPreflightResult records a successful scratch-storage preflight. It
// contains no corpus content, vectors, or secrets.
type ScratchPreflightResult struct {
	Directory string
	Elapsed   time.Duration
}

// PreflightScratch proves that the verifier can create, initialize, write to,
// and clean up its disk-backed scratch relation in directory. It reuses the
// exact verification-relation mechanics so a success predicts late-phase
// verification success rather than only directory writability.
//
// PreflightScratch creates directory (mode 0700) if it does not exist. It
// fails closed when directory is empty so callers cannot silently fall back
// to Go's default temporary directory. The probe leaves no files behind on
// success.
func PreflightScratch(ctx context.Context, directory string) (ScratchPreflightResult, error) {
	started := time.Now()
	if strings.TrimSpace(directory) == "" {
		return ScratchPreflightResult{}, errors.New("bundle verification scratch directory is required")
	}
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return ScratchPreflightResult{}, errors.Wrap(err, "create bundle verification scratch directory")
	}
	relation, err := openVerificationRelation(ctx, directory)
	if err != nil {
		return ScratchPreflightResult{}, err
	}
	// Exercise a real write through the same chunk-identity admission path
	// used during verification, then commit and clean up. This proves the
	// relation file is writable by the current process, SQLite can open and
	// initialize it, a transaction can commit, and the file removes cleanly.
	if err := relation.addChunk(ctx, "preflight-chunk", "preflight-document", 0, "preflight-digest"); err != nil {
		_ = relation.closeAndRemove()
		return ScratchPreflightResult{}, errors.Wrap(err, "preflight bundle verification write")
	}
	if err := relation.finishChunks(ctx); err != nil {
		_ = relation.closeAndRemove()
		return ScratchPreflightResult{}, errors.Wrap(err, "preflight bundle verification commit")
	}
	if err := relation.closeAndRemove(); err != nil {
		return ScratchPreflightResult{}, errors.Wrap(err, "preflight bundle verification cleanup")
	}
	return ScratchPreflightResult{Directory: directory, Elapsed: time.Since(started)}, nil
}
