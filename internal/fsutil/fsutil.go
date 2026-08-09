package fsutil

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// AtomicWriteOptions controls file publication. Zero modes use secure defaults:
// 0700 for created directories and 0600 for the destination file.
type AtomicWriteOptions struct {
	DirectoryMode os.FileMode
	FileMode      os.FileMode
	TempPattern   string
}

// AtomicWrite publishes data by writing and syncing a sibling temporary file,
// renaming it over path, and syncing the containing directory.
func AtomicWrite(ctx context.Context, path string, data []byte, options AtomicWriteOptions) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	directoryMode := options.DirectoryMode
	if directoryMode == 0 {
		directoryMode = 0o700
	}
	fileMode := options.FileMode
	if fileMode == 0 {
		fileMode = 0o600
	}
	pattern := options.TempPattern
	if pattern == "" {
		pattern = ".atomic-*"
	}
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, directoryMode); err != nil {
		return fmt.Errorf("create parent directory: %w", err)
	}
	temporary, err := os.CreateTemp(directory, pattern)
	if err != nil {
		return fmt.Errorf("create temporary file: %w", err)
	}
	temporaryName := temporary.Name()
	defer func() {
		_ = temporary.Close()
		_ = os.Remove(temporaryName)
	}()
	if err := temporary.Chmod(fileMode); err != nil {
		return fmt.Errorf("set file permissions: %w", err)
	}
	if _, err := temporary.Write(data); err != nil {
		return fmt.Errorf("write temporary file: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		return fmt.Errorf("sync temporary file: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close temporary file: %w", err)
	}
	if err := os.Rename(temporaryName, path); err != nil {
		return fmt.Errorf("publish file: %w", err)
	}
	if err := SyncDirectory(directory); err != nil {
		return fmt.Errorf("sync parent directory: %w", err)
	}
	return nil
}

// SyncDirectory syncs directory metadata to durable storage.
func SyncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	if err := directory.Sync(); err != nil {
		_ = directory.Close()
		return err
	}
	return directory.Close()
}

// JoinWithin lexically joins a non-empty relative path beneath root. It does
// not resolve symlinks and is therefore containment validation, not a defense
// against a hostile filesystem that can modify symlinks concurrently.
func JoinWithin(root, relative string) (string, error) {
	if strings.TrimSpace(root) == "" {
		return "", fmt.Errorf("root path is required")
	}
	if relative == "" || filepath.IsAbs(relative) {
		return "", fmt.Errorf("path must be non-empty and relative")
	}
	clean := filepath.Clean(relative)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes root")
	}
	root = filepath.Clean(root)
	path := filepath.Join(root, clean)
	relativeToRoot, err := filepath.Rel(root, path)
	if err != nil || relativeToRoot == ".." || strings.HasPrefix(relativeToRoot, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes root")
	}
	return path, nil
}

// DirectorySize returns the total Lstat size of non-directory entries beneath
// root. Symlinks are counted as entries and are not followed.
func DirectorySize(ctx context.Context, root string) (int64, error) {
	var total int64
	err := filepath.WalkDir(root, func(_ string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		total += info.Size()
		return nil
	})
	if err != nil {
		return 0, err
	}
	return total, nil
}
