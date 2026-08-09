package rag

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestUsageAddPreservesMissingAndReportedZero(t *testing.T) {
	t.Parallel()
	var total Usage
	total.Add(Usage{})
	require.Equal(t, Usage{}, total)
	zero := int64(0)
	total.Add(Usage{InputTokens: &zero})
	require.NotNil(t, total.InputTokens)
	require.Zero(t, *total.InputTokens)
	require.Nil(t, total.OutputTokens)
	three := int64(3)
	total.Add(Usage{InputTokens: &three})
	require.Equal(t, int64(3), *total.InputTokens)
	total.Add(Usage{CachedInputTokens: &three})
	require.Equal(t, int64(3), *total.CachedInputTokens)
	total.Add(Usage{ReasoningTokens: &three})
	require.Equal(t, int64(3), *total.ReasoningTokens)
}

func TestUsageAddDoesNotAliasInputs(t *testing.T) {
	t.Parallel()
	input := int64(2)
	var total Usage
	total.Add(Usage{InputTokens: &input})
	input = 9
	require.Equal(t, int64(2), *total.InputTokens)
	total.Add(Usage{InputTokens: int64Pointer(1)})
	require.Equal(t, int64(9), input)
}

func TestUsageCloneDoesNotAliasPointers(t *testing.T) {
	t.Parallel()
	input := int64(2)
	cost := 0.5
	original := Usage{InputTokens: &input, CostUSD: &cost}
	clone := original.Clone()
	*clone.InputTokens = 9
	*clone.CostUSD = 1.5
	require.Equal(t, int64(2), *original.InputTokens)
	require.Equal(t, 0.5, *original.CostUSD)
}

func TestUsageAccumulatorSnapshotDoesNotChange(t *testing.T) {
	t.Parallel()
	var accumulator UsageAccumulator
	one := int64(1)
	var group sync.WaitGroup
	for range 20 {
		group.Add(1)
		go func() {
			defer group.Done()
			accumulator.Add(Usage{InputTokens: &one})
		}()
	}
	group.Wait()
	first := accumulator.Snapshot()
	accumulator.Add(Usage{InputTokens: &one})
	require.Equal(t, int64(20), *first.InputTokens)
	require.Equal(t, int64(21), *accumulator.Snapshot().InputTokens)
}

func int64Pointer(value int64) *int64 {
	return &value
}
