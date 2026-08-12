package flow

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/go-go-golems/flowkit/execution"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/require"
)

// chunkStep is the standalone per-item step batched tests repair with; its
// identity mirrors what a per-item run would use, so repairs share entries.
func chunkStep(calls *atomic.Int64) Step[string, string] {
	return Step[string, string]{
		Name: "chunk",
		Identity: Identity[string]{
			Kind:    "test-chunk",
			Version: "v1",
			Key:     func(item string) ([]byte, error) { return []byte(item), nil },
		},
		Policy: Policy{Workers: 2},
		Do: func(_ context.Context, item string) (string, error) {
			if calls != nil {
				calls.Add(1)
			}
			return "repaired:" + item, nil
		},
	}
}

// groupsOf splits indexes into fixed-size groups.
func groupsOf(size int) func(items []string) [][]int {
	return func(items []string) [][]int {
		groups := [][]int{}
		for start := 0; start < len(items); start += size {
			end := min(start+size, len(items))
			group := make([]int, 0, end-start)
			for index := start; index < end; index++ {
				group = append(group, index)
			}
			groups = append(groups, group)
		}
		return groups
	}
}

// splitJSON parses {"0":"text",...} group responses.
func splitJSON(raw string, _ []string) (map[int]string, error) {
	parsed := map[string]string{}
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil, AsDataError(err)
	}
	results := map[int]string{}
	for key, value := range parsed {
		var position int
		if _, err := fmt.Sscanf(key, "%d", &position); err != nil {
			continue
		}
		results[position] = value
	}
	return results, nil
}

func TestBatchedHappyPathUsesOnlyGroupCalls(t *testing.T) {
	var repairCalls, groupCalls atomic.Int64
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 2},
		Group:  groupsOf(2),
		DoAll: func(_ context.Context, group []string) (string, error) {
			groupCalls.Add(1)
			response := map[string]string{}
			for position, item := range group {
				response[fmt.Sprintf("%d", position)] = "batched:" + item
			}
			data, err := json.Marshal(response)
			return string(data), err
		},
		Split: splitJSON,
	}
	step := Batched(chunkStep(&repairCalls), spec)
	items := []string{"a", "b", "c", "d", "e"}
	results, report, err := Run(context.Background(), step, items, Options{})
	require.NoError(t, err)
	for index, item := range items {
		require.Equal(t, "batched:"+item, results[index].Value)
	}
	require.Equal(t, int64(3), groupCalls.Load())
	require.Equal(t, int64(0), repairCalls.Load(), "a healthy run needs no repairs")
	require.Equal(t, 3, report.Step("chunk-groups").Items)
	require.Equal(t, 0, report.Step("chunk").Items)
}

func TestBatchedOuterResultHookSeesSplitAndRepairedItems(t *testing.T) {
	var repairCalls atomic.Int64
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 1},
		Group:  groupsOf(3),
		DoAll: func(_ context.Context, group []string) (string, error) {
			return fmt.Sprintf(`{"0":"batched:%s","2":"batched:%s"}`, group[0], group[2]), nil
		},
		Split: splitJSON,
	}
	step := Batched(chunkStep(&repairCalls), spec)
	seen := map[int]string{}
	step.OnResult = func(_ context.Context, index int, value string, _ execution.CacheOutcome) error {
		seen[index] = value
		return nil
	}
	results, _, err := Run(context.Background(), step, []string{"a", "b", "c"}, Options{})
	require.NoError(t, err)
	require.Equal(t, "repaired:b", results[1].Value)
	require.Equal(t, map[int]string{0: "batched:a", 1: "repaired:b", 2: "batched:c"}, seen)
	require.Equal(t, int64(1), repairCalls.Load())
}

func TestBatchedOuterResultHookFailureFailsRun(t *testing.T) {
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 1},
		Group:  groupsOf(1),
		DoAll: func(_ context.Context, _ []string) (string, error) {
			return `{"0":"batched:a"}`, nil
		},
		Split: splitJSON,
	}
	step := Batched(chunkStep(nil), spec)
	step.OnResult = func(context.Context, int, string, execution.CacheOutcome) error {
		return errors.New("artifact writer failed")
	}
	_, _, err := Run(context.Background(), step, []string{"a"}, Options{})
	require.ErrorContains(t, err, "result hook")
}

func TestBatchedRepairsMissingAndUnparseableItems(t *testing.T) {
	var repairCalls atomic.Int64
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 1},
		Group:  groupsOf(3),
		DoAll: func(_ context.Context, group []string) (string, error) {
			// The "model" drops position 1 of every group.
			response := map[string]string{}
			for position, item := range group {
				if position == 1 {
					continue
				}
				response[fmt.Sprintf("%d", position)] = "batched:" + item
			}
			data, err := json.Marshal(response)
			return string(data), err
		},
		Split: splitJSON,
	}
	step := Batched(chunkStep(&repairCalls), spec)
	items := []string{"a", "b", "c", "d", "e", "f"}
	results, report, err := Run(context.Background(), step, items, Options{})
	require.NoError(t, err)
	require.Equal(t, "batched:a", results[0].Value)
	require.Equal(t, "repaired:b", results[1].Value)
	require.Equal(t, "batched:c", results[2].Value)
	require.Equal(t, "repaired:e", results[4].Value)
	require.Equal(t, int64(2), repairCalls.Load())
	require.Equal(t, 2, report.Step("chunk").Items)
}

func TestBatchedRepairSharesCacheWithStandaloneRuns(t *testing.T) {
	// The campaign invariant: a repair must hit the cache entry a standalone
	// per-item run already stored, because the identities byte-match.
	store := NewMemoryStore()
	var standaloneCalls, repairCalls atomic.Int64

	standalone := chunkStep(&standaloneCalls)
	_, _, err := Run(context.Background(), standalone, []string{"b"}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, int64(1), standaloneCalls.Load())

	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 1},
		Group:  groupsOf(2),
		DoAll: func(_ context.Context, group []string) (string, error) {
			return "{}", nil // the model returns nothing usable
		},
		Split: splitJSON,
	}
	step := Batched(chunkStep(&repairCalls), spec)
	results, report, err := Run(context.Background(), step, []string{"a", "b"}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, "repaired:a", results[0].Value)
	require.Equal(t, "repaired:b", results[1].Value)
	require.Equal(t, int64(1), repairCalls.Load(), `"b" must be a cache hit from the standalone run`)
	require.Equal(t, 1, report.Step("chunk").Hits)
	require.Equal(t, 1, report.Step("chunk").Misses)
}

func TestBatchedGroupCallsAreDurableUnderTheirOwnIdentity(t *testing.T) {
	store := NewMemoryStore()
	var groupCalls atomic.Int64
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 1},
		Identity: Identity[[]string]{
			Kind:    "test-chunk-groups",
			Version: "v1",
			Key: func(group []string) ([]byte, error) {
				return []byte(strings.Join(group, "\x00")), nil
			},
		},
		Group: groupsOf(2),
		DoAll: func(_ context.Context, group []string) (string, error) {
			groupCalls.Add(1)
			response := map[string]string{}
			for position, item := range group {
				response[fmt.Sprintf("%d", position)] = "batched:" + item
			}
			data, err := json.Marshal(response)
			return string(data), err
		},
		Split: splitJSON,
	}
	step := Batched(chunkStep(nil), spec)
	_, _, err := Run(context.Background(), step, []string{"a", "b", "c"}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, int64(2), groupCalls.Load())

	_, report, err := Run(context.Background(), step, []string{"a", "b", "c"}, Options{Store: store})
	require.NoError(t, err)
	require.Equal(t, int64(2), groupCalls.Load(), "group replay must be free")
	require.Equal(t, 2, report.Step("chunk-groups").Hits)
}

func TestBatchedSplitErrorSendsWholeGroupToRepair(t *testing.T) {
	var repairCalls atomic.Int64
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 1},
		Group:  groupsOf(2),
		DoAll: func(_ context.Context, _ []string) (string, error) {
			return "not json at all", nil
		},
		Split: splitJSON,
	}
	step := Batched(chunkStep(&repairCalls), spec)
	results, _, err := Run(context.Background(), step, []string{"a", "b"}, Options{})
	require.NoError(t, err)
	require.Equal(t, "repaired:a", results[0].Value)
	require.Equal(t, "repaired:b", results[1].Value)
	require.Equal(t, int64(2), repairCalls.Load())
}

func TestBatchedQuarantinedGroupRoutesItemsToRepair(t *testing.T) {
	var repairCalls atomic.Int64
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 1, OnError: Quarantine, Retry: fastRetry(2)},
		Group:  groupsOf(2),
		DoAll: func(_ context.Context, _ []string) (string, error) {
			return "", errors.New("read: connection timed out")
		},
		Split: splitJSON,
	}
	step := Batched(chunkStep(&repairCalls), spec)
	results, report, err := Run(context.Background(), step, []string{"a", "b"}, Options{})
	require.NoError(t, err)
	require.Equal(t, "repaired:a", results[0].Value)
	require.Equal(t, "repaired:b", results[1].Value)
	require.Equal(t, 1, report.Step("chunk-groups").Quarantined)
	require.Equal(t, 1, report.Step("chunk-groups").Retries)
	require.Equal(t, int64(2), repairCalls.Load())
}

func TestBatchedGroupFailureUnderFailFastKillsRun(t *testing.T) {
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 1},
		Group:  groupsOf(2),
		DoAll: func(_ context.Context, _ []string) (string, error) {
			return "", errors.New("status=401: bad key")
		},
		Split: splitJSON,
	}
	step := Batched(chunkStep(nil), spec)
	_, _, err := Run(context.Background(), step, []string{"a", "b"}, Options{})
	require.Error(t, err)
	require.Contains(t, err.Error(), "chunk-groups")
}

func TestBatchedSharedResourceDrawsOneBudget(t *testing.T) {
	// Group calls and repairs share one admission resource: the
	// BatchedCallCeiling arithmetic (one possible group + repair per item).
	resource := Resource{Name: "family-calls", Ceiling: 4, Budget: 2}
	repair := chunkStep(nil)
	repair.Policy.Admission = []Resource{resource}
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 1, Admission: []Resource{resource}},
		Group:  groupsOf(2),
		DoAll: func(_ context.Context, _ []string) (string, error) {
			return "{}", nil // force both items into repair
		},
		Split: splitJSON,
	}
	step := Batched(repair, spec)
	_, _, err := Run(context.Background(), step, []string{"a", "b"}, Options{})
	require.Error(t, err, "1 group call + 2 repairs cannot fit a shared budget of 2")
	require.Contains(t, err.Error(), `resource "family-calls" admission refused`)
}

func TestBatchedValidatesGroupIndexes(t *testing.T) {
	badSpecs := map[string]func([]string) [][]int{
		"out of range": func(items []string) [][]int { return [][]int{{0, 99}} },
		"duplicated":   func(items []string) [][]int { return [][]int{{0, 0}} },
	}
	for name, group := range badSpecs {
		spec := BatchSpec[string, string]{
			Policy: Policy{},
			Group:  group,
			DoAll:  func(_ context.Context, _ []string) (string, error) { return "{}", nil },
			Split:  splitJSON,
		}
		_, _, err := Run(context.Background(), Batched(chunkStep(nil), spec), []string{"a", "b"}, Options{})
		require.Error(t, err, name)
	}
}

func TestBatchedUncoveredIndexesGoToRepair(t *testing.T) {
	var repairCalls atomic.Int64
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 1},
		Group:  func(items []string) [][]int { return [][]int{{0}} }, // ignores item 1
		DoAll: func(_ context.Context, group []string) (string, error) {
			return `{"0":"batched:` + group[0] + `"}`, nil
		},
		Split: splitJSON,
	}
	step := Batched(chunkStep(&repairCalls), spec)
	results, _, err := Run(context.Background(), step, []string{"a", "b"}, Options{})
	require.NoError(t, err)
	require.Equal(t, "batched:a", results[0].Value)
	require.Equal(t, "repaired:b", results[1].Value)
	require.Equal(t, int64(1), repairCalls.Load())
}

func TestBatchedInsidePipeActsAsBarrierStage(t *testing.T) {
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 2},
		Group:  groupsOf(2),
		DoAll: func(_ context.Context, group []string) (string, error) {
			response := map[string]string{}
			for position, item := range group {
				response[fmt.Sprintf("%d", position)] = "batched:" + item
			}
			data, err := json.Marshal(response)
			return string(data), err
		},
		Split: splitJSON,
	}
	batched := Batched(chunkStep(nil), spec)
	upper := Step[string, string]{
		Name:   "upper",
		Policy: Policy{Workers: 2},
		Do: func(_ context.Context, item string) (string, error) {
			return strings.ToUpper(item), nil
		},
	}
	results, report, err := Run(context.Background(), Pipe2(batched, upper), []string{"a", "b", "c"}, Options{})
	require.NoError(t, err)
	require.Equal(t, "BATCHED:A", results[0].Value)
	require.Equal(t, "BATCHED:C", results[2].Value)
	require.Equal(t, 3, report.Step("upper").Items)
	require.Equal(t, 2, report.Step("chunk-groups").Items)
}

func TestBatchedBarrierRestoresOriginalInputOrder(t *testing.T) {
	release := make(chan struct{})
	upstream := Step[string, string]{
		Name: "reordering-upstream", Policy: Policy{Workers: 2},
		Do: func(_ context.Context, item string) (string, error) {
			if item == "a" {
				<-release
			} else {
				close(release)
			}
			return item, nil
		},
	}
	var received []string
	spec := BatchSpec[string, string]{
		Policy: Policy{Workers: 1},
		Group: func(items []string) [][]int {
			indexes := make([]int, len(items))
			for i := range items {
				indexes[i] = i
			}
			return [][]int{indexes}
		},
		DoAll: func(_ context.Context, group []string) (string, error) {
			received = append([]string(nil), group...)
			response := map[string]string{"0": group[0], "1": group[1]}
			data, err := json.Marshal(response)
			return string(data), err
		},
		Split: splitJSON,
	}
	batched := Batched(chunkStep(nil), spec)
	_, _, err := Run(t.Context(), Pipe2(upstream, batched), []string{"a", "b"}, Options{})
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b"}, received)
}
