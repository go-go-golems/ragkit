package execution

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestFileCacheRoundTrip(t *testing.T) {
	t.Parallel()

	cache, err := NewFileCache(FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatalf("NewFileCache() error = %v", err)
	}
	key, err := NewKey("embed", "v1", map[string]string{"document": "doc-1", "model": "model-a"})
	if err != nil {
		t.Fatalf("NewKey() error = %v", err)
	}
	const wantInputDigest = "38b43f573fb669b340bc09197e84d7481936e1b09a62618ec52ed9045fcc5683"
	if key.InputDigest != wantInputDigest {
		t.Fatalf("NewKey() input digest = %q, want %q", key.InputDigest, wantInputDigest)
	}
	want := cachedFixture{ID: "doc-1", Values: []float32{1, 2, 3}}
	if err := cache.Store(context.Background(), key, want); err != nil {
		t.Fatalf("Store() error = %v", err)
	}
	path, err := cache.path(key)
	if err != nil {
		t.Fatalf("path() error = %v", err)
	}
	const wantKeyDigest = "f1f6cf68b7bcbba1263d260fef5bdd69f72005eef733ddd2043eb16a224e14be"
	if got := filepath.Base(filepath.Dir(path)); got != wantKeyDigest[:2] {
		t.Fatalf("cache shard = %q, want %q", got, wantKeyDigest[:2])
	}
	if got := filepath.Base(path); got != wantKeyDigest+".json" {
		t.Fatalf("cache file = %q, want %q", got, wantKeyDigest+".json")
	}

	var got cachedFixture
	found, err := cache.Load(context.Background(), key, &got)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if !found {
		t.Fatal("Load() found = false, want true")
	}
	if got.ID != want.ID || len(got.Values) != len(want.Values) {
		t.Fatalf("Load() = %+v, want %+v", got, want)
	}
}

func TestFileCacheFailsClosedOnCorruption(t *testing.T) {
	t.Parallel()

	cache, err := NewFileCache(FileCacheOptions{Directory: t.TempDir()})
	if err != nil {
		t.Fatalf("NewFileCache() error = %v", err)
	}
	key, err := NewKey("embed", "v1", "doc-1")
	if err != nil {
		t.Fatalf("NewKey() error = %v", err)
	}
	if err := cache.Store(context.Background(), key, cachedFixture{ID: "doc-1"}); err != nil {
		t.Fatalf("Store() error = %v", err)
	}
	path, err := cache.path(key)
	if err != nil {
		t.Fatalf("path() error = %v", err)
	}
	if err := os.WriteFile(path, []byte(`{"schema_version":"wrong"}`), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var result cachedFixture
	_, err = cache.Load(context.Background(), key, &result)
	if !errors.Is(err, ErrCorruptCache) {
		t.Fatalf("Load() error = %v, want ErrCorruptCache", err)
	}
}

func TestFileCacheLeavesNoTemporaryFiles(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	cache, err := NewFileCache(FileCacheOptions{Directory: root})
	if err != nil {
		t.Fatalf("NewFileCache() error = %v", err)
	}
	key, err := NewKey("embed", "v1", "doc-1")
	if err != nil {
		t.Fatalf("NewKey() error = %v", err)
	}
	if err := cache.Store(context.Background(), key, cachedFixture{ID: "doc-1"}); err != nil {
		t.Fatalf("Store() error = %v", err)
	}
	if err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && filepath.Ext(path) != ".json" {
			t.Errorf("unexpected cache file %s", path)
		}
		return nil
	}); err != nil {
		t.Fatalf("WalkDir() error = %v", err)
	}
}

func TestFileCacheRejectsOversizedEntry(t *testing.T) {
	t.Parallel()

	cache, err := NewFileCache(FileCacheOptions{Directory: t.TempDir(), MaxEntryBytes: 64})
	if err != nil {
		t.Fatalf("NewFileCache() error = %v", err)
	}
	key, err := NewKey("embed", "v1", "doc-1")
	if err != nil {
		t.Fatalf("NewKey() error = %v", err)
	}
	if err := cache.Store(context.Background(), key, make([]byte, 128)); err == nil {
		t.Fatal("Store() error = nil, want maximum entry error")
	}
}

func TestNewKeyIsStableAcrossMapOrder(t *testing.T) {
	t.Parallel()

	left, err := NewKey("embed", "v1", map[string]int{"a": 1, "b": 2})
	if err != nil {
		t.Fatalf("NewKey(left) error = %v", err)
	}
	right, err := NewKey("embed", "v1", map[string]int{"b": 2, "a": 1})
	if err != nil {
		t.Fatalf("NewKey(right) error = %v", err)
	}
	leftJSON, _ := json.Marshal(left)
	rightJSON, _ := json.Marshal(right)
	if string(leftJSON) != string(rightJSON) {
		t.Fatalf("keys differ: %s != %s", leftJSON, rightJSON)
	}
}

type cachedFixture struct {
	ID     string    `json:"id"`
	Values []float32 `json:"values"`
}
