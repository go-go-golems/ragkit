package flow

import (
	"context"
	"fmt"

	"github.com/go-go-golems/ragkit/execution"
)

// BatchSpec configures batching-with-repair: many items share one provider
// call, and whatever the group response misses is repaired per item.
type BatchSpec[I, O any] struct {
	// Name is the group-call step's report name. Empty selects the repair
	// step's name suffixed "-groups".
	Name string
	// Identity optionally makes raw group responses durable under their own
	// keyspace. Kind == "" leaves group calls uncached — appropriate when
	// DoAll already runs through its own durable step.
	Identity Identity[[]I]
	// Policy governs the group calls (workers, admission, retry, failure
	// mode). Name a shared admission resource in both Policy and the repair
	// step's policy to draw group calls and repairs from one budget, the
	// BatchedCallCeiling arithmetic.
	Policy Policy
	// Group partitions items into index groups. Every index must appear at
	// most once; uncovered indexes go straight to repair.
	Group func(items []I) [][]int
	// DoAll performs one provider call for a whole group and returns the
	// raw response.
	DoAll func(ctx context.Context, group []I) (string, error)
	// Split parses per-item results out of a raw group response, keyed by
	// position within the group. Missing positions are repaired; a Split
	// error sends the whole group to repair (the response was transported
	// fine — it is the batch that failed as data).
	Split func(raw string, group []I) (map[int]O, error)
}

// Batched builds a step that runs items through grouped calls plus per-item
// repairs, preserving the campaign's cache invariant: the repair step's
// Identity must byte-match the standalone per-item step so repairs and
// standalone runs share cache entries. Group-call failures that do not kill
// the run (per the group Policy's FailureMode) route every item of the
// group to repair.
func Batched[I, O any](repair Step[I, O], spec BatchSpec[I, O]) Step[I, O] {
	name := spec.Name
	if name == "" {
		name = repair.Name + "-groups"
	}
	step := Step[I, O]{
		Name:       name,
		Policy:     spec.Policy,
		extraPlans: policyPlans(repair.Name, repair.Policy),
	}
	step.override = func(ctx context.Context, items []I, o Options, onResult func(context.Context, int, O, execution.CacheOutcome) error) ([]Result[O], Report, error) {
		return runBatched(ctx, repair, spec, name, items, o, onResult)
	}
	return step
}

func runBatched[I, O any](
	ctx context.Context,
	repair Step[I, O],
	spec BatchSpec[I, O],
	name string,
	items []I,
	o Options,
	onResult func(context.Context, int, O, execution.CacheOutcome) error,
) ([]Result[O], Report, error) {
	report := Report{}
	if spec.Group == nil || spec.DoAll == nil || spec.Split == nil {
		return nil, report, fmt.Errorf("batched step %q needs Group, DoAll, and Split", name)
	}
	if repair.Do == nil && repair.override == nil && repair.stages == nil {
		return nil, report, fmt.Errorf("batched step %q needs a runnable repair step", name)
	}

	groups := spec.Group(items)
	covered := make([]bool, len(items))
	for groupIndex, group := range groups {
		for _, itemIndex := range group {
			if itemIndex < 0 || itemIndex >= len(items) {
				return nil, report, fmt.Errorf(
					"batched step %q: group %d references item %d outside [0,%d)",
					name, groupIndex, itemIndex, len(items),
				)
			}
			if covered[itemIndex] {
				return nil, report, fmt.Errorf(
					"batched step %q: item %d appears in more than one group",
					name, itemIndex,
				)
			}
			covered[itemIndex] = true
		}
	}

	groupStep := Step[[]I, string]{
		Name:     name,
		Identity: spec.Identity,
		Policy:   spec.Policy,
		Do:       spec.DoAll,
	}
	groupInputs := make([][]I, len(groups))
	for groupIndex, group := range groups {
		members := make([]I, len(group))
		for position, itemIndex := range group {
			members[position] = items[itemIndex]
		}
		groupInputs[groupIndex] = members
	}
	rawResponses, groupReport, err := Run(ctx, groupStep, groupInputs, o)
	report.merge(groupReport)
	if err != nil {
		return nil, report, err
	}

	results := make([]Result[O], len(items))
	missing := []int{}
	for itemIndex, isCovered := range covered {
		if !isCovered {
			missing = append(missing, itemIndex)
		}
	}
	for groupIndex, raw := range rawResponses {
		group := groups[groupIndex]
		if raw.Quarantined != nil || raw.Skipped {
			missing = append(missing, group...)
			continue
		}
		parsed, splitErr := spec.Split(raw.Value, groupInputs[groupIndex])
		for position, itemIndex := range group {
			if splitErr != nil {
				missing = append(missing, itemIndex)
				continue
			}
			value, ok := parsed[position]
			if !ok {
				missing = append(missing, itemIndex)
				continue
			}
			results[itemIndex] = Result[O]{Value: value}
			if onResult != nil {
				if err := onResult(ctx, itemIndex, value, execution.CacheOutcome{}); err != nil {
					return nil, report, fmt.Errorf("step %q item %d: result hook: %w", name, itemIndex, err)
				}
			}
		}
	}

	if len(missing) == 0 {
		return results, report, nil
	}
	repairItems := make([]I, len(missing))
	for position, itemIndex := range missing {
		repairItems[position] = items[itemIndex]
	}
	repaired, repairReport, err := Run(ctx, repair, repairItems, o)
	report.merge(repairReport)
	if err != nil {
		return nil, report, err
	}
	for position, itemIndex := range missing {
		result := repaired[position]
		if result.Quarantined != nil {
			adjusted := *result.Quarantined
			adjusted.Index = itemIndex
			result.Quarantined = &adjusted
		}
		results[itemIndex] = result
		if result.Quarantined == nil && !result.Skipped && onResult != nil {
			if err := onResult(ctx, itemIndex, result.Value, result.Cache); err != nil {
				return nil, report, fmt.Errorf("step %q item %d: result hook: %w", name, itemIndex, err)
			}
		}
	}
	return results, report, nil
}
