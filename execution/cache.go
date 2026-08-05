package execution

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/go-go-golems/ragkit/digest"
	"github.com/go-go-golems/ragkit/internal/fsutil"
	"github.com/go-go-golems/ragkit/internal/jsonutil"
)

const cacheEntrySchema = "rag-ttc-execution-cache/v1"

// ErrCorruptCache identifies an entry that exists but fails validation. Cache
// corruption fails closed instead of silently repeating expensive work.
var ErrCorruptCache = errors.New("execution cache entry is corrupt")

// Key identifies one deterministic item result. Step names the operation,
// Version changes whenever its semantics change, and InputDigest covers every
// input that can affect the result.
type Key struct {
	Step        string `json:"step"`
	Version     string `json:"version"`
	InputDigest string `json:"input_digest"`
}

// NewKey hashes a JSON-serializable input into a deterministic cache key.
// Callers must include model, prompt, provider, and normalization identities in
// input whenever they affect the result.
func NewKey(step, version string, input any) (Key, error) {
	if step == "" {
		return Key{}, fmt.Errorf("cache step is required")
	}
	if version == "" {
		return Key{}, fmt.Errorf("cache step version is required")
	}
	data, err := json.Marshal(input)
	if err != nil {
		return Key{}, fmt.Errorf("marshal cache input: %w", err)
	}
	return Key{
		Step:        step,
		Version:     version,
		InputDigest: digest.Bytes(data),
	}, nil
}

func (key Key) validate() error {
	if key.Step == "" || key.Version == "" {
		return fmt.Errorf("cache key step and version are required")
	}
	if len(key.InputDigest) != sha256.Size*2 {
		return fmt.Errorf("cache key input digest is invalid")
	}
	if _, err := hex.DecodeString(key.InputDigest); err != nil {
		return fmt.Errorf("cache key input digest is invalid: %w", err)
	}
	return nil
}

func (key Key) digest() (string, error) {
	if err := key.validate(); err != nil {
		return "", err
	}
	data, err := json.Marshal(key)
	if err != nil {
		return "", err
	}
	return digest.Bytes(data), nil
}

// Cache stores completed item results. Load unmarshals into target and returns
// false only when no entry exists. Invalid existing entries return an error.
type Cache interface {
	Load(context.Context, Key, any) (bool, error)
	Store(context.Context, Key, any) error
}

// FileCacheOptions configures a process-local JSON cache.
type FileCacheOptions struct {
	Directory     string
	MaxEntryBytes int64
}

// FileCache stores one validated result envelope per key.
type FileCache struct {
	directory     string
	maxEntryBytes int64
}

var _ Cache = (*FileCache)(nil)

// NewFileCache creates a durable item cache. The default maximum entry size is
// 16 MiB.
func NewFileCache(options FileCacheOptions) (*FileCache, error) {
	if options.Directory == "" {
		return nil, fmt.Errorf("cache directory is required")
	}
	maxEntryBytes := options.MaxEntryBytes
	if maxEntryBytes == 0 {
		maxEntryBytes = 16 << 20
	}
	if maxEntryBytes < 1 {
		return nil, fmt.Errorf("cache maximum entry bytes must be positive")
	}
	if err := os.MkdirAll(options.Directory, 0o700); err != nil {
		return nil, fmt.Errorf("create cache directory: %w", err)
	}
	return &FileCache{
		directory:     options.Directory,
		maxEntryBytes: maxEntryBytes,
	}, nil
}

type cacheEnvelope struct {
	SchemaVersion string          `json:"schema_version"`
	Key           Key             `json:"key"`
	ValueDigest   string          `json:"value_digest"`
	Value         json.RawMessage `json:"value"`
}

func (cache *FileCache) path(key Key) (string, error) {
	keyDigest, err := key.digest()
	if err != nil {
		return "", err
	}
	return filepath.Join(cache.directory, keyDigest[:2], keyDigest+".json"), nil
}

// Load validates the schema, full key, value digest, and result JSON.
func (cache *FileCache) Load(ctx context.Context, key Key, target any) (bool, error) {
	if cache == nil {
		return false, fmt.Errorf("file cache is nil")
	}
	if target == nil {
		return false, fmt.Errorf("cache load target is required")
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	path, err := cache.path(key)
	if err != nil {
		return false, err
	}
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("stat cache entry: %w", err)
	}
	if info.Size() > cache.maxEntryBytes {
		return false, fmt.Errorf("%w: entry exceeds maximum size", ErrCorruptCache)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return false, fmt.Errorf("read cache entry: %w", err)
	}
	envelope, err := jsonutil.DecodeStrict[cacheEnvelope](data)
	if err != nil {
		return false, fmt.Errorf("%w: decode envelope: %v", ErrCorruptCache, err)
	}
	if envelope.SchemaVersion != cacheEntrySchema ||
		envelope.Key != key ||
		envelope.ValueDigest != digest.Bytes(envelope.Value) {
		return false, fmt.Errorf("%w: envelope validation failed", ErrCorruptCache)
	}
	if err := jsonutil.DecodeStrictInto(envelope.Value, target); err != nil {
		return false, fmt.Errorf("%w: decode value: %v", ErrCorruptCache, err)
	}
	return true, nil
}

// Store atomically publishes a complete result envelope. Results are visible
// only after the value and containing directory have been synced.
func (cache *FileCache) Store(ctx context.Context, key Key, value any) error {
	if cache == nil {
		return fmt.Errorf("file cache is nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	path, err := cache.path(key)
	if err != nil {
		return err
	}
	valueData, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("marshal cache value: %w", err)
	}
	envelope := cacheEnvelope{
		SchemaVersion: cacheEntrySchema,
		Key:           key,
		ValueDigest:   digest.Bytes(valueData),
		Value:         valueData,
	}
	data, err := json.Marshal(envelope)
	if err != nil {
		return fmt.Errorf("marshal cache envelope: %w", err)
	}
	if int64(len(data)) > cache.maxEntryBytes {
		return fmt.Errorf("cache entry is %d bytes, maximum is %d", len(data), cache.maxEntryBytes)
	}

	if err := fsutil.AtomicWrite(ctx, path, data, fsutil.AtomicWriteOptions{TempPattern: ".entry-*"}); err != nil {
		return fmt.Errorf("publish cache entry: %w", err)
	}
	return nil
}
