package flow

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/go-go-golems/ragkit/execution"
)

// Preflight enables the monetary gate over every declared resource plan. A
// nil Preflight in Options skips the ceiling-coverage and cost refusals
// (callers that state their own refusal arithmetic still get budgets and
// caching); runtime budgets enforce regardless.
type Preflight struct {
	MaxEstimatedUSD float64
	AllowUnpriced   bool
	AllowPartial    bool
}

// Options carries the run-scoped collaborators: the durable store, the
// event ledger, the monetary preflight, and optional per-resource rate
// limiters composed after the finite budgets.
type Options struct {
	// Store is the durability seam; nil means an uncached run.
	Store Store
	// Ledger optionally journals run events; a ledger error fails the run.
	Ledger Ledger
	// Preflight optionally gates plan coverage and estimated cost before
	// item one.
	Preflight *Preflight
	// Rates maps resource names to rate limiters, composed after the finite
	// budget. Extra names are ignored (the map is typically shared across
	// many steps).
	Rates map[string]execution.Limiter

	// env shares budgets across the stages and sub-runs of one Run call.
	env *runEnv
}

// Share returns Options whose budgets, preflight arithmetic, and admission
// state are shared by every Run call made with the returned value: two steps
// declaring the same resource name then draw from one budget, exactly like
// the old harness-wide budgets. Without Share, each top-level Run owns its
// budgets. Share is idempotent: already-shared Options keep their state.
func (o Options) Share() Options {
	if o.env == nil {
		o.env = newRunEnv(o.Rates, o.Preflight)
	}
	return o
}

// Declare registers admission plans before any step runs, with the same
// fail-closed validation and refusal arithmetic steps get. Use it on shared
// Options when non-step consumers (legacy per-call caches) must draw from
// the same budgets, or when refusals must happen before expensive
// preparation. The step name only attributes refusal messages.
func (o Options) Declare(step string, resources ...Resource) error {
	if o.env == nil {
		return fmt.Errorf("Declare needs shared Options: call Share() first")
	}
	plans := make([]stagePlan, 0, len(resources))
	for _, resource := range resources {
		plans = append(plans, stagePlan{step: step, resource: resource})
	}
	return o.env.ensure(plans)
}

// Limiter returns the composed budget+rate limiter for a declared resource,
// for consumers that spend outside flow steps. Nil when the resource was
// never declared.
func (o Options) Limiter(name string) execution.Limiter {
	if o.env == nil {
		return nil
	}
	return o.env.limiter(name)
}

// Snapshots returns current spend for every declared resource, in the shape
// harnesses serialize into run directories.
func (o Options) Snapshots() map[string]execution.BudgetSnapshot {
	if o.env == nil {
		return nil
	}
	return o.env.allSnapshots()
}

// Cost returns the accumulated monetary preflight over every declared plan.
func (o Options) Cost() execution.CostPreflight {
	if o.env == nil {
		return execution.CostPreflight{}
	}
	return o.env.cost()
}

// stagePlan attributes one admission resource to its declaring step.
type stagePlan struct {
	step     string
	resource Resource
}

type policySpec struct {
	step   string
	policy Policy
}

// runEnv owns one budget and composed limiter per resource name for a whole
// Run, including nested runs (Batched group calls and repairs share their
// caller's budgets when they share resource names). It also owns the
// fail-closed preflight: every plan is validated exactly once, before any
// item that could consume it.
type runEnv struct {
	gate *Preflight

	mutex        sync.Mutex
	plans        map[string]execution.ResourcePlan
	owners       map[string]string
	budgets      map[string]*execution.Budget
	limiters     map[string]execution.Limiter
	rates        map[string]execution.Limiter
	estimatedUSD float64
	unpriced     []string
}

func newRunEnv(rates map[string]execution.Limiter, gate *Preflight) *runEnv {
	return &runEnv{
		gate:     gate,
		plans:    map[string]execution.ResourcePlan{},
		owners:   map[string]string{},
		budgets:  map[string]*execution.Budget{},
		limiters: map[string]execution.Limiter{},
		rates:    rates,
	}
}

// ensure validates and registers plans, creating one budget per resource
// name. A resource declared by two steps must be declared identically and
// then shares one budget — that is how a batched family's group calls and
// repairs draw from one ceiling. Refusals happen here, before item one,
// and state their arithmetic.
func (env *runEnv) ensure(plans []stagePlan) error {
	env.mutex.Lock()
	defer env.mutex.Unlock()
	if env.gate != nil && (math.IsNaN(env.gate.MaxEstimatedUSD) || math.IsInf(env.gate.MaxEstimatedUSD, 0) || env.gate.MaxEstimatedUSD < 0) {
		return fmt.Errorf("preflight maximum USD must be finite and non-negative")
	}
	type pendingPlan struct {
		stage   stagePlan
		plan    execution.ResourcePlan
		budget  *execution.Budget
		limiter execution.Limiter
	}
	pending := make([]pendingPlan, 0, len(plans))
	prospectivePlans := make(map[string]execution.ResourcePlan, len(env.plans)+len(plans))
	prospectiveOwners := make(map[string]string, len(env.owners)+len(plans))
	for name, plan := range env.plans {
		prospectivePlans[name] = plan
		prospectiveOwners[name] = env.owners[name]
	}
	prospectiveUSD := env.estimatedUSD
	prospectiveUnpriced := append([]string(nil), env.unpriced...)
	for _, plan := range plans {
		name := plan.resource.Name
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("step %q declares an admission resource without a name", plan.step)
		}
		converted := plan.resource.plan()
		reference := plan.resource.Ceiling == 0 && plan.resource.Budget == 0 && plan.resource.UnitUSD == nil
		if existing, ok := prospectivePlans[name]; ok {
			// A zero-valued Resource{Name: ...} is a reference to an earlier
			// full declaration: the step draws from that budget.
			if reference || resourcePlansEqual(existing, converted) {
				continue
			}
			return fmt.Errorf(
				"resource %q is declared differently by steps %q and %q",
				name, prospectiveOwners[name], plan.step,
			)
		}
		// A zero-valued resource with no earlier declaration becomes a zero
		// plan: it admits nothing, and the first spend refuses with the
		// resource name — a mistyped reference fails loudly, not silently.
		if plan.resource.Ceiling < 0 {
			return fmt.Errorf("step %q: resource %q ceiling must be non-negative", plan.step, name)
		}
		if plan.resource.Budget < 0 {
			return fmt.Errorf("step %q: resource %q budget must be non-negative", plan.step, name)
		}
		if plan.resource.UnitUSD != nil && (math.IsNaN(*plan.resource.UnitUSD) || math.IsInf(*plan.resource.UnitUSD, 0) || *plan.resource.UnitUSD < 0) {
			return fmt.Errorf("step %q: resource %q unit price must be finite and non-negative", plan.step, name)
		}
		if env.gate != nil && !env.gate.AllowPartial && plan.resource.Budget < plan.resource.Ceiling {
			return fmt.Errorf(
				"step %q: resource %q budget %d cannot cover the stated ceiling of %d calls; raise the budget to at least %d (cache hits are free)",
				plan.step, name, plan.resource.Budget, plan.resource.Ceiling, plan.resource.Ceiling,
			)
		}
		if plan.resource.Ceiling > 0 {
			if plan.resource.UnitUSD == nil {
				prospectiveUnpriced = append(prospectiveUnpriced, name)
			} else {
				prospectiveUSD += float64(plan.resource.Ceiling) * *plan.resource.UnitUSD
			}
		}
		budget, err := execution.NewBudget(plan.resource.Budget)
		if err != nil {
			return fmt.Errorf("step %q: create resource %q budget: %w", plan.step, name, err)
		}
		prospectivePlans[name] = converted
		prospectiveOwners[name] = plan.step
		pending = append(pending, pendingPlan{
			stage: plan, plan: converted, budget: budget,
			limiter: execution.Chain(budget, env.rates[name]),
		})
	}
	if env.gate != nil {
		if len(prospectiveUnpriced) > 0 && !env.gate.AllowUnpriced {
			return fmt.Errorf("preflight refused the plans: pricing unavailable for %s", strings.Join(prospectiveUnpriced, ", "))
		}
		if prospectiveUSD > env.gate.MaxEstimatedUSD {
			return fmt.Errorf(
				"preflight refused the plans: estimated provider cost %.6f exceeds maximum USD %.6f",
				prospectiveUSD, env.gate.MaxEstimatedUSD,
			)
		}
	}
	for _, admitted := range pending {
		name := admitted.stage.resource.Name
		env.plans[name] = admitted.plan
		env.owners[name] = admitted.stage.step
		env.budgets[name] = admitted.budget
		env.limiters[name] = admitted.limiter
		log.Debug().
			Str("step", admitted.stage.step).
			Str("resource", name).
			Int("ceiling", admitted.stage.resource.Ceiling).
			Int("budget", admitted.stage.resource.Budget).
			Bool("priced", admitted.stage.resource.UnitUSD != nil).
			Msg("flow resource plan admitted")
	}
	env.estimatedUSD = prospectiveUSD
	env.unpriced = prospectiveUnpriced
	return nil
}

func resourcePlansEqual(left, right execution.ResourcePlan) bool {
	if left.Name != right.Name || left.Ceiling != right.Ceiling || left.Budget != right.Budget {
		return false
	}
	if left.UnitUSD == nil || right.UnitUSD == nil {
		return left.UnitUSD == nil && right.UnitUSD == nil
	}
	return *left.UnitUSD == *right.UnitUSD
}

func retryDelay(delay, cap time.Duration) time.Duration {
	if cap <= 0 {
		return 0
	}
	if delay < 0 {
		delay = 0
	}
	if delay > cap {
		delay = cap
	}
	jitterLimit := delay / 2
	jitter := time.Duration(rand.Int63n(int64(jitterLimit) + 1))
	if delay >= cap-jitter {
		return cap
	}
	return delay + jitter
}

func nextRetryDelay(delay, cap time.Duration) time.Duration {
	if cap <= 0 {
		return 0
	}
	if delay >= cap {
		return cap
	}
	if delay <= 0 {
		return 0
	}
	if delay > cap-delay {
		return cap
	}
	return delay * 2
}

func (env *runEnv) limiter(name string) execution.Limiter {
	env.mutex.Lock()
	defer env.mutex.Unlock()
	return env.limiters[name]
}

// admit reserves every named resource as one transaction. Per-resource
// limiters are themselves reservable chains, so a refusal rolls back budgets
// and rate tokens acquired for earlier resource names before provider work.
func (env *runEnv) admit(ctx context.Context, names []string, units int) (string, error) {
	reservations := make([]execution.Reservation, 0, len(names))
	rollback := func() {
		for index := len(reservations) - 1; index >= 0; index-- {
			reservations[index].Rollback()
		}
	}
	for _, name := range names {
		limiter := env.limiter(name)
		if limiter == nil {
			rollback()
			return name, fmt.Errorf("resource has no admitted limiter")
		}
		if reservable, ok := limiter.(execution.ReservableLimiter); ok {
			reservation, err := reservable.Reserve(ctx, units)
			if err != nil {
				rollback()
				return name, err
			}
			reservations = append(reservations, reservation)
			continue
		}
		if err := limiter.Wait(ctx, units); err != nil {
			rollback()
			return name, err
		}
	}
	for _, reservation := range reservations {
		reservation.Commit()
	}
	return "", nil
}

func (env *runEnv) plan(name string) execution.ResourcePlan {
	env.mutex.Lock()
	defer env.mutex.Unlock()
	return env.plans[name]
}

func (env *runEnv) allSnapshots() map[string]execution.BudgetSnapshot {
	env.mutex.Lock()
	defer env.mutex.Unlock()
	snapshots := make(map[string]execution.BudgetSnapshot, len(env.budgets))
	for name, budget := range env.budgets {
		snapshots[name] = budget.Snapshot()
	}
	return snapshots
}

func (env *runEnv) cost() execution.CostPreflight {
	env.mutex.Lock()
	defer env.mutex.Unlock()
	return execution.CostPreflight{
		EstimatedUSD:  env.estimatedUSD,
		MissingPrices: append([]string(nil), env.unpriced...),
	}
}

func (env *runEnv) snapshot(names []string) map[string]execution.BudgetSnapshot {
	if len(names) == 0 {
		return nil
	}
	env.mutex.Lock()
	defer env.mutex.Unlock()
	snapshots := make(map[string]execution.BudgetSnapshot, len(names))
	for _, name := range names {
		snapshots[name] = env.budgets[name].Snapshot()
	}
	return snapshots
}

// Run executes one step (or composed pipeline) over items. Results[i]
// corresponds to items[i], always. The report covers every stage that ran
// and is returned best-effort alongside errors.
func Run[I, O any](ctx context.Context, s Step[I, O], items []I, o Options) ([]Result[O], Report, error) {
	stages := stagesOf(s)
	for _, stage := range stages {
		if err := validatePolicy(stage.name, stage.policy); err != nil {
			return nil, Report{}, err
		}
		for _, nested := range stage.extraPolicies {
			if err := validatePolicy(nested.step, nested.policy); err != nil {
				return nil, Report{}, err
			}
		}
	}
	if o.env == nil {
		o.env = newRunEnv(o.Rates, o.Preflight)
	}
	if err := o.env.ensure(collectPlans(stages)); err != nil {
		return nil, Report{}, err
	}
	if s.override != nil {
		return s.override(ctx, items, o, s.OnResult)
	}

	inputs := make([]erasedItem, len(items))
	for index, item := range items {
		inputs[index] = erasedItem{index: index, value: item}
	}
	outputs, report, err := runStages(ctx, stages, inputs, o)
	if err != nil {
		return nil, report, err
	}
	results := make([]Result[O], len(items))
	for index, output := range outputs {
		results[index] = Result[O]{
			Cache:       output.outcome,
			Quarantined: output.quarantined,
			Skipped:     output.skipped,
		}
		if output.quarantined == nil && !output.skipped && output.value != nil {
			results[index].Value = output.value.(O)
		}
	}
	return results, report, nil
}

// collectPlans gathers every stage's admission plans.
func collectPlans(stages []stageSpec) []stagePlan {
	plans := []stagePlan{}
	for _, stage := range stages {
		plans = append(plans, stage.plans...)
	}
	return plans
}

// policyPlans attributes a policy's admission resources to a step name.
func policyPlans(step string, policy Policy) []stagePlan {
	plans := make([]stagePlan, 0, len(policy.Admission))
	for _, resource := range policy.Admission {
		plans = append(plans, stagePlan{step: step, resource: resource})
	}
	return plans
}

// erasedItem is one item moving through a type-erased pipeline. A flagged
// item (quarantined or skipped) bypasses every later stage unchanged.
type erasedItem struct {
	index       int
	value       any
	outcome     execution.CacheOutcome
	quarantined *ItemError
	skipped     bool
}

func (item erasedItem) flagged() bool { return item.quarantined != nil || item.skipped }

// stageSpec is one pipeline stage: its identity for reports, its admission
// plans for the shared preflight, and a builder for its runner.
type stageSpec struct {
	name          string
	barrier       bool
	policy        Policy
	extraPolicies []policySpec
	plans         []stagePlan
	build         func(o Options) (stageRunner, error)
}

// stageRunner processes a stream of erased items. run must not close the
// channels — the driver owns them — and returns the first fatal error.
type stageRunner interface {
	run(ctx context.Context, in <-chan erasedItem, out chan<- erasedItem) error
	report() Report
}

// stagesOf returns the flattened stage list of a step: its own composed
// stages, or itself as a single stage.
func stagesOf[I, O any](s Step[I, O]) []stageSpec {
	if s.stages != nil {
		return s.stages
	}
	return []stageSpec{stageOfStep(s)}
}

// stageOfStep wraps one typed step as a pipeline stage. A step with a custom
// engine (Batched) becomes a barrier stage running that engine over the
// clean items.
func stageOfStep[I, O any](s Step[I, O]) stageSpec {
	plans := policyPlans(s.Name, s.Policy)
	plans = append(plans, s.extraPlans...)
	if s.override != nil {
		return stageSpec{
			name:          s.Name,
			barrier:       true,
			policy:        s.Policy,
			extraPolicies: s.extraPolicies,
			plans:         plans,
			build: func(o Options) (stageRunner, error) {
				return &overrideStageRunner[I, O]{step: s, options: o}, nil
			},
		}
	}
	return stageSpec{
		name:          s.Name,
		barrier:       s.Barrier,
		policy:        s.Policy,
		extraPolicies: s.extraPolicies,
		plans:         plans,
		build: func(o Options) (stageRunner, error) {
			return newTypedRunner(s, o)
		},
	}
}

// progressInterval throttles the info-level progress line long runs emit.
// Runs shorter than one interval never log; a 17k-item build says where it
// is twice a minute instead of being silent for an hour.
var progressInterval = 30 * time.Second

// logProgress periodically logs every running stage's counters until stop
// closes. Snapshots come from the runners' own reports, so the line shows
// exactly what the final report will count.
func logProgress(stop <-chan struct{}, total int, runners []stageRunner) {
	ticker := time.NewTicker(progressInterval)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			for _, runner := range runners {
				for name, step := range runner.report().Steps {
					log.Info().
						Str("step", name).
						Int("items", step.Items).
						Int("total", total).
						Int("hits", step.Hits).
						Int("misses", step.Misses).
						Int("work_calls", step.WorkCalls).
						Int("retries", step.Retries).
						Int("quarantined", step.Quarantined).
						Msg("flow progress")
				}
			}
		}
	}
}

// runStages drives the pipeline: a feeder, one runner per stage connected by
// channels (per-item streaming), and an order-restoring collector.
func runStages(ctx context.Context, stages []stageSpec, inputs []erasedItem, o Options) ([]erasedItem, Report, error) {
	runners := make([]stageRunner, len(stages))
	for index, stage := range stages {
		runner, err := stage.build(o)
		if err != nil {
			return nil, Report{}, fmt.Errorf("stage %q: %w", stage.name, err)
		}
		runners[index] = runner
	}

	progressStop := make(chan struct{})
	defer close(progressStop)
	go logProgress(progressStop, len(inputs), runners)

	group, groupContext := errgroup.WithContext(ctx)
	feed := make(chan erasedItem)
	group.Go(func() error {
		defer close(feed)
		for _, input := range inputs {
			select {
			case <-groupContext.Done():
				return groupContext.Err()
			case feed <- input:
			}
		}
		return nil
	})

	current := (<-chan erasedItem)(feed)
	for index, runner := range runners {
		out := make(chan erasedItem, 16)
		in := current
		stage := stages[index]
		boundRunner := runner
		group.Go(func() error {
			defer close(out)
			if err := boundRunner.run(groupContext, in, out); err != nil {
				return fmt.Errorf("stage %q: %w", stage.name, err)
			}
			return nil
		})
		current = out
	}

	results := make([]erasedItem, len(inputs))
	final := current
	group.Go(func() error {
		for item := range final {
			results[item.index] = item
		}
		return nil
	})

	err := group.Wait()
	report := Report{}
	for _, runner := range runners {
		report.merge(runner.report())
	}
	if err != nil {
		return nil, report, err
	}
	return results, report, nil
}

// overrideStageRunner embeds a custom-engine step (Batched) into a pipeline
// as a barrier stage: it drains the stream, runs the engine over the clean
// items, and re-emits everything position-aligned.
type overrideStageRunner[I, O any] struct {
	step    Step[I, O]
	options Options
	mutex   sync.Mutex
	rep     Report
}

func (runner *overrideStageRunner[I, O]) run(ctx context.Context, in <-chan erasedItem, out chan<- erasedItem) error {
	var flagged []erasedItem
	var clean []erasedItem
	for item := range in {
		if item.flagged() {
			flagged = append(flagged, item)
			continue
		}
		clean = append(clean, item)
	}
	sort.Slice(clean, func(left, right int) bool { return clean[left].index < clean[right].index })
	if err := ctx.Err(); err != nil {
		return err
	}
	items := make([]I, len(clean))
	for position, item := range clean {
		typed, ok := item.value.(I)
		if !ok {
			return fmt.Errorf("item %d has type %T, stage expects %T", item.index, item.value, *new(I))
		}
		items[position] = typed
	}
	results, report, err := Run(ctx, runner.step, items, runner.options)
	runner.mutex.Lock()
	runner.rep.merge(report)
	runner.mutex.Unlock()
	if err != nil {
		return err
	}
	emit := func(item erasedItem) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case out <- item:
			return nil
		}
	}
	for position, result := range results {
		item := erasedItem{
			index:       clean[position].index,
			outcome:     result.Cache,
			quarantined: result.Quarantined,
			skipped:     result.Skipped,
		}
		if result.Quarantined == nil && !result.Skipped {
			item.value = result.Value
		}
		if item.quarantined != nil {
			adjusted := *item.quarantined
			adjusted.Index = item.index
			item.quarantined = &adjusted
		}
		if err := emit(item); err != nil {
			return err
		}
	}
	for _, item := range flagged {
		if err := emit(item); err != nil {
			return err
		}
	}
	return nil
}

func (runner *overrideStageRunner[I, O]) report() Report {
	runner.mutex.Lock()
	defer runner.mutex.Unlock()
	return runner.rep
}

// inflightCall shares one key's execution between duplicate items so a key
// never runs twice in one process — duplicate nondeterministic work would
// desynchronize the run's output from the stored entry.
type inflightCall struct {
	done        chan struct{}
	value       any
	state       execution.CacheState
	quarantined *ItemError
	skipped     bool
	err         error
}

// typedRunner is the per-stage engine for a plain step: cache, admission,
// retry, failure policy, in-flight dedup, and report counting.
type typedRunner[I, O any] struct {
	step       Step[I, O]
	options    Options
	classifier Classifier
	backoff    Backoff
	attempts   int
	workers    int
	resources  []string

	mutex    sync.Mutex
	counts   StepReport
	meters   Meters
	inflight map[string]*inflightCall
}

func newTypedRunner[I, O any](s Step[I, O], o Options) (*typedRunner[I, O], error) {
	if s.Name == "" {
		return nil, fmt.Errorf("every step needs a name")
	}
	if s.Do == nil {
		return nil, fmt.Errorf("step %q has no Do function", s.Name)
	}
	if s.Meter != nil && s.AttemptMeter != nil {
		return nil, fmt.Errorf("step %q cannot set both Meter and AttemptMeter", s.Name)
	}
	classifier := s.Policy.Retry.Class
	if classifier == nil {
		classifier = DefaultClassifier
	}
	attempts := s.Policy.Retry.Attempts
	if attempts < 1 {
		attempts = 1
	}
	workers := s.Policy.Workers
	if workers < 1 {
		workers = 1
	}
	resources := make([]string, 0, len(s.Policy.Admission))
	for _, resource := range s.Policy.Admission {
		resources = append(resources, resource.Name)
	}
	return &typedRunner[I, O]{
		step:       s,
		options:    o,
		classifier: classifier,
		backoff:    s.Policy.Retry.Backoff.withDefaults(),
		attempts:   attempts,
		workers:    workers,
		resources:  resources,
		meters:     Meters{},
		inflight:   map[string]*inflightCall{},
	}, nil
}

func validatePolicy(step string, policy Policy) error {
	if policy.Retry.Attempts < 0 {
		return fmt.Errorf("step %q retry attempts must not be negative", step)
	}
	switch policy.OnError {
	case FailFast, Quarantine, Skip:
	default:
		return fmt.Errorf("step %q has unknown failure mode %d", step, policy.OnError)
	}
	seen := make(map[string]struct{}, len(policy.Admission))
	for _, resource := range policy.Admission {
		name := strings.TrimSpace(resource.Name)
		if _, duplicate := seen[name]; duplicate {
			return fmt.Errorf("step %q declares admission resource %q more than once", step, name)
		}
		seen[name] = struct{}{}
	}
	return nil
}

func (runner *typedRunner[I, O]) run(ctx context.Context, in <-chan erasedItem, out chan<- erasedItem) error {
	source := in
	if runner.step.Barrier {
		var buffered []erasedItem
		for item := range in {
			buffered = append(buffered, item)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		barrier := make(chan erasedItem)
		go func() {
			defer close(barrier)
			for _, item := range buffered {
				select {
				case <-ctx.Done():
					return
				case barrier <- item:
				}
			}
		}()
		source = barrier
	}

	group, groupContext := errgroup.WithContext(ctx)
	for range runner.workers {
		group.Go(func() error {
			for item := range source {
				if item.flagged() {
					select {
					case <-groupContext.Done():
						return groupContext.Err()
					case out <- item:
					}
					continue
				}
				typed, ok := item.value.(I)
				if !ok {
					return fmt.Errorf("item %d has type %T, step %q expects %T", item.index, item.value, runner.step.Name, *new(I))
				}
				processed, err := runner.process(groupContext, item.index, typed)
				if err != nil {
					return err
				}
				select {
				case <-groupContext.Done():
					return groupContext.Err()
				case out <- processed:
				}
			}
			return nil
		})
	}
	return group.Wait()
}

func (runner *typedRunner[I, O]) report() Report {
	runner.mutex.Lock()
	defer runner.mutex.Unlock()
	counts := runner.counts
	if len(runner.meters) > 0 {
		counts.Meters = Meters{}
		counts.Meters.Add(runner.meters)
	}
	counts.Spend = runner.options.env.snapshot(runner.resources)
	return Report{Steps: map[string]StepReport{runner.step.Name: counts}}
}

func (runner *typedRunner[I, O]) count(update func(*StepReport)) {
	runner.mutex.Lock()
	update(&runner.counts)
	runner.mutex.Unlock()
}

// notify invokes the step's OnResult hook for one successful value.
func (runner *typedRunner[I, O]) notify(ctx context.Context, index int, value O, outcome execution.CacheOutcome) error {
	if runner.step.OnResult == nil {
		return nil
	}
	if err := runner.step.OnResult(ctx, index, value, outcome); err != nil {
		return fmt.Errorf("step %q item %d: result hook: %w", runner.step.Name, index, err)
	}
	return nil
}

func (runner *typedRunner[I, O]) event(ctx context.Context, event Event) error {
	if runner.options.Ledger == nil {
		return nil
	}
	event.Step = runner.step.Name
	if err := runner.options.Ledger.Event(ctx, event); err != nil {
		return fmt.Errorf("step %q: ledger event: %w", runner.step.Name, err)
	}
	return nil
}

// process resolves one item to its outcome. The returned error is fatal to
// the whole run; everything else becomes a result.
func (runner *typedRunner[I, O]) process(ctx context.Context, index int, item I) (erasedItem, error) {
	runner.count(func(counts *StepReport) { counts.Items++ })
	cached := runner.step.Identity.Kind != "" && runner.options.Store != nil
	if !cached {
		return runner.work(ctx, index, item, nil, "")
	}

	key, err := runner.step.Key(item)
	if err != nil {
		return erasedItem{}, err
	}
	digestValue, err := keyDigest(key)
	if err != nil {
		return erasedItem{}, fmt.Errorf("step %q item %d: %w", runner.step.Name, index, err)
	}

	runner.mutex.Lock()
	if call, ok := runner.inflight[digestValue]; ok {
		runner.mutex.Unlock()
		return runner.follow(ctx, index, call, digestValue)
	}
	call := &inflightCall{done: make(chan struct{})}
	runner.inflight[digestValue] = call
	runner.mutex.Unlock()

	result, err := runner.lead(ctx, index, item, key, digestValue)
	call.err = err
	call.value = result.value
	call.state = result.outcome.State
	call.quarantined = result.quarantined
	call.skipped = result.skipped
	close(call.done)
	return result, err
}

// follow waits for the leading duplicate and replicates its outcome.
func (runner *typedRunner[I, O]) follow(ctx context.Context, index int, call *inflightCall, digestValue string) (erasedItem, error) {
	select {
	case <-ctx.Done():
		return erasedItem{}, ctx.Err()
	case <-call.done:
	}
	if call.err != nil {
		return erasedItem{}, call.err
	}
	result := erasedItem{index: index, value: call.value, skipped: call.skipped}
	if call.state != "" {
		result.outcome = execution.CacheOutcome{KeyDigest: digestValue, State: call.state}
	}
	eventType := EventStored
	switch {
	case call.quarantined != nil:
		quarantined := *call.quarantined
		quarantined.Index = index
		result.quarantined = &quarantined
		runner.count(func(counts *StepReport) { counts.Misses++; counts.Quarantined++ })
		eventType = EventQuarantined
	case call.skipped:
		runner.count(func(counts *StepReport) { counts.Misses++; counts.Skipped++ })
		eventType = EventSkipped
	case call.state == execution.CacheHit:
		runner.count(func(counts *StepReport) { counts.Hits++ })
		eventType = EventHit
	default:
		runner.count(func(counts *StepReport) { counts.Misses++ })
	}
	if err := runner.event(ctx, Event{Index: index, Type: eventType}); err != nil {
		return erasedItem{}, err
	}
	if result.quarantined == nil && !result.skipped {
		value, _ := result.value.(O)
		if err := runner.notify(ctx, index, value, result.outcome); err != nil {
			return erasedItem{}, err
		}
	}
	return result, nil
}

// lead executes one deduplicated key: load, admit, retry, store.
func (runner *typedRunner[I, O]) lead(ctx context.Context, index int, item I, key execution.Key, digestValue string) (erasedItem, error) {
	var loaded O
	found, err := runner.options.Store.Load(ctx, key, &loaded)
	if err != nil {
		return erasedItem{}, fmt.Errorf("step %q item %d: load cache entry: %w", runner.step.Name, index, err)
	}
	if found {
		runner.count(func(counts *StepReport) { counts.Hits++ })
		if err := runner.event(ctx, Event{Index: index, Type: EventHit}); err != nil {
			return erasedItem{}, err
		}
		outcome := execution.CacheOutcome{KeyDigest: digestValue, State: execution.CacheHit}
		if err := runner.notify(ctx, index, loaded, outcome); err != nil {
			return erasedItem{}, err
		}
		return erasedItem{index: index, value: loaded, outcome: outcome}, nil
	}
	runner.count(func(counts *StepReport) { counts.Misses++ })
	return runner.work(ctx, index, item, &key, digestValue)
}

// work admits, retries, and stores one fresh call. Every Do invocation,
// retries included, obtains its own admission because Policy.Admission prices
// work calls rather than logical items. Cache hits never enter this path.
func (runner *typedRunner[I, O]) work(ctx context.Context, index int, item I, key *execution.Key, digestValue string) (erasedItem, error) {
	delay := runner.backoff.Base
	var lastErr error
	var lastClass ErrorClass
	attemptsMade := 0
	for attempt := 1; attempt <= runner.attempts; attempt++ {
		if attempt > 1 {
			jittered := retryDelay(delay, runner.backoff.Cap)
			select {
			case <-ctx.Done():
				return erasedItem{}, ctx.Err()
			case <-time.After(jittered):
			}
			delay = nextRetryDelay(delay, runner.backoff.Cap)
		}
		if name, err := runner.options.env.admit(ctx, runner.resources, 1); err != nil {
			if errors.Is(err, execution.ErrBudgetExceeded) {
				plan := runner.options.env.plan(name)
				return erasedItem{}, fmt.Errorf(
					"step %q item %d attempt %d: resource %q admission refused: %w; stated ceiling %d, admitted budget %d — cache hits are free",
					runner.step.Name, index, attempt, name, err, plan.Ceiling, plan.Budget,
				)
			}
			return erasedItem{}, fmt.Errorf("step %q item %d attempt %d: wait for resource %q: %w", runner.step.Name, index, attempt, name, err)
		}
		attemptsMade = attempt
		value, err := runner.step.Do(ctx, item)
		runner.count(func(counts *StepReport) { counts.WorkCalls++ })
		if runner.step.AttemptMeter != nil {
			metered := runner.step.AttemptMeter(value, err)
			runner.mutex.Lock()
			runner.meters.Add(metered)
			runner.mutex.Unlock()
		}
		if err == nil {
			return runner.success(ctx, index, value, key, digestValue)
		}
		lastErr = err
		if ctx.Err() != nil {
			return erasedItem{}, fmt.Errorf("step %q item %d: %w", runner.step.Name, index, err)
		}
		lastClass = runner.classifier.Classify(err)
		if !lastClass.valid() {
			return erasedItem{}, fmt.Errorf("step %q item %d: classifier returned unknown error class %d", runner.step.Name, index, lastClass)
		}
		log.Debug().
			Str("step", runner.step.Name).
			Int("item", index).
			Int("attempt", attempt).
			Str("class", lastClass.String()).
			Err(err).
			Msg("flow work call failed")
		if lastClass == Transient && attempt < runner.attempts {
			runner.count(func(counts *StepReport) {
				counts.Retries++
				if counts.RetriesByClass == nil {
					counts.RetriesByClass = map[string]int{}
				}
				counts.RetriesByClass[lastClass.String()]++
			})
			if err := runner.event(ctx, Event{
				Index: index, Type: EventRetry, Class: lastClass.String(),
				Attempt: attempt, Error: err.Error(),
			}); err != nil {
				return erasedItem{}, err
			}
			continue
		}
		break
	}
	return runner.fail(ctx, index, lastErr, lastClass, attemptsMade)
}

// success meters and stores one fresh result.
func (runner *typedRunner[I, O]) success(ctx context.Context, index int, value O, key *execution.Key, digestValue string) (erasedItem, error) {
	if runner.step.Meter != nil {
		metered := runner.step.Meter(value)
		runner.mutex.Lock()
		runner.meters.Add(metered)
		runner.mutex.Unlock()
	}
	result := erasedItem{index: index, value: value}
	eventType := EventDone
	if key != nil {
		// Once expensive work succeeds, finish its local atomic commit even
		// if a sibling item has just canceled the run.
		if err := runner.options.Store.Store(context.WithoutCancel(ctx), *key, value); err != nil {
			return erasedItem{}, fmt.Errorf("step %q item %d: store cache entry: %w", runner.step.Name, index, err)
		}
		runner.count(func(counts *StepReport) { counts.Stored++ })
		result.outcome = execution.CacheOutcome{KeyDigest: digestValue, State: execution.CacheStored}
		eventType = EventStored
	}
	if err := runner.event(ctx, Event{Index: index, Type: eventType}); err != nil {
		return erasedItem{}, err
	}
	if err := runner.notify(ctx, index, value, result.outcome); err != nil {
		return erasedItem{}, err
	}
	return result, nil
}

// fail resolves one exhausted item error to its destiny: kill the run,
// quarantine the item as data, or drop it with a count.
func (runner *typedRunner[I, O]) fail(ctx context.Context, index int, cause error, class ErrorClass, attempts int) (erasedItem, error) {
	mode := runner.step.Policy.OnError
	if class == Fatal || mode == FailFast {
		return erasedItem{}, fmt.Errorf("step %q item %d [%s]: %w", runner.step.Name, index, class, cause)
	}
	if mode == Skip {
		runner.count(func(counts *StepReport) { counts.Skipped++ })
		if err := runner.event(ctx, Event{
			Index: index, Type: EventSkipped, Class: class.String(), Error: cause.Error(),
		}); err != nil {
			return erasedItem{}, err
		}
		return erasedItem{index: index, skipped: true}, nil
	}
	record := &ItemError{
		Step:     runner.step.Name,
		Index:    index,
		Class:    class,
		Attempts: attempts,
		Message:  cause.Error(),
	}
	runner.count(func(counts *StepReport) { counts.Quarantined++ })
	if err := runner.event(ctx, Event{
		Index: index, Type: EventQuarantined, Class: class.String(), Error: cause.Error(),
	}); err != nil {
		return erasedItem{}, err
	}
	return erasedItem{index: index, quarantined: record}, nil
}
