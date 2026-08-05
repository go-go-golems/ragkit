package rag

import "sync"

// Add adds every reported field from value. A nil field remains missing unless
// at least one added value reports it. Reported zero remains non-nil.
func (usage *Usage) Add(value Usage) {
	if usage == nil {
		return
	}
	usage.InputTokens = sumInt64(usage.InputTokens, value.InputTokens)
	usage.CachedInputTokens = sumInt64(
		usage.CachedInputTokens, value.CachedInputTokens,
	)
	usage.OutputTokens = sumInt64(usage.OutputTokens, value.OutputTokens)
	usage.ReasoningTokens = sumInt64(usage.ReasoningTokens, value.ReasoningTokens)
	usage.EmbeddingTokens = sumInt64(usage.EmbeddingTokens, value.EmbeddingTokens)
	usage.CostUSD = sumFloat64(usage.CostUSD, value.CostUSD)
}

// UsageAccumulator aggregates provider usage safely across goroutines.
type UsageAccumulator struct {
	mutex sync.Mutex
	usage Usage
}

// Add incorporates value.
func (accumulator *UsageAccumulator) Add(value Usage) {
	if accumulator == nil {
		return
	}
	accumulator.mutex.Lock()
	defer accumulator.mutex.Unlock()
	accumulator.usage.Add(value)
}

// Snapshot returns a pointer-independent copy of accumulated usage.
func (accumulator *UsageAccumulator) Snapshot() Usage {
	if accumulator == nil {
		return Usage{}
	}
	accumulator.mutex.Lock()
	defer accumulator.mutex.Unlock()
	var copy Usage
	copy.Add(accumulator.usage)
	return copy
}

func sumInt64(left, right *int64) *int64 {
	if right == nil {
		if left == nil {
			return nil
		}
		value := *left
		return &value
	}
	value := *right
	if left != nil {
		value += *left
	}
	return &value
}

func sumFloat64(left, right *float64) *float64 {
	if right == nil {
		if left == nil {
			return nil
		}
		value := *left
		return &value
	}
	value := *right
	if left != nil {
		value += *left
	}
	return &value
}
