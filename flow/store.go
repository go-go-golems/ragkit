package flow

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/go-go-golems/ragkit/execution"
)

// Store is the swappable durability seam. It carries exactly the execution
// cache contract: Load unmarshals into target and returns false only when no
// entry exists (an invalid existing entry is an error — corruption fails
// closed), and Store atomically publishes a complete result. The canonical
// implementation is *execution.FileCache (content-addressed JSON envelopes
// on disk, the provider-steps cache every recorded run replays from);
// swapping the durable mechanism means implementing these two methods —
// steps, pipelines, and reports never change.
type Store interface {
	Load(ctx context.Context, key execution.Key, target any) (bool, error)
	Store(ctx context.Context, key execution.Key, value any) error
}

var _ Store = (*execution.FileCache)(nil)

// MemoryStore is an in-process Store for tests and ephemeral runs. It proves
// the durability swap: the same steps run unchanged against it, they just
// stop surviving the process.
type MemoryStore struct {
	mutex   sync.RWMutex
	entries map[string][]byte
}

var _ Store = (*MemoryStore)(nil)

// NewMemoryStore creates an empty in-process store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{entries: map[string][]byte{}}
}

// Load implements Store.
func (store *MemoryStore) Load(ctx context.Context, key execution.Key, target any) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	digestValue, err := keyDigest(key)
	if err != nil {
		return false, err
	}
	store.mutex.RLock()
	data, ok := store.entries[digestValue]
	store.mutex.RUnlock()
	if !ok {
		return false, nil
	}
	if err := json.Unmarshal(data, target); err != nil {
		return false, fmt.Errorf("decode memory store entry: %w", err)
	}
	return true, nil
}

// Store implements Store.
func (store *MemoryStore) Store(ctx context.Context, key execution.Key, value any) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	digestValue, err := keyDigest(key)
	if err != nil {
		return err
	}
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("marshal memory store value: %w", err)
	}
	store.mutex.Lock()
	store.entries[digestValue] = data
	store.mutex.Unlock()
	return nil
}

// Len returns the number of stored entries.
func (store *MemoryStore) Len() int {
	store.mutex.RLock()
	defer store.mutex.RUnlock()
	return len(store.entries)
}
