package execution

import (
	"fmt"
	"math"
	"strings"
)

// ResourcePlan declares the maximum work and admitted budget for one named
// integer resource. A nil UnitUSD means its price is unknown.
type ResourcePlan struct {
	Name    string   `json:"name"`
	Ceiling int      `json:"ceiling"`
	Budget  int      `json:"budget"`
	UnitUSD *float64 `json:"unit_usd,omitempty"`
}

// CostPreflight records conservative ceiling pricing and unknown prices.
type CostPreflight struct {
	EstimatedUSD  float64  `json:"estimated_usd"`
	MissingPrices []string `json:"missing_prices,omitempty"`
}

// ValidateResourcePlans validates resource identities, budgets, and pricing
// against an explicit monetary ceiling.
func ValidateResourcePlans(
	plans []ResourcePlan,
	maximumUSD float64,
	allowUnpriced bool,
	allowPartial bool,
) (CostPreflight, error) {
	preflight := CostPreflight{}
	if math.IsNaN(maximumUSD) || math.IsInf(maximumUSD, 0) || maximumUSD < 0 {
		return preflight, fmt.Errorf("maximum USD must be finite and non-negative")
	}
	seen := make(map[string]bool, len(plans))
	for index, plan := range plans {
		if strings.TrimSpace(plan.Name) == "" {
			return preflight, fmt.Errorf("resource plan %d: name is required", index)
		}
		if seen[plan.Name] {
			return preflight, fmt.Errorf("resource plan %q is duplicated", plan.Name)
		}
		seen[plan.Name] = true
		if plan.Ceiling < 0 {
			return preflight, fmt.Errorf("resource %q ceiling must be non-negative", plan.Name)
		}
		if plan.Budget < 0 {
			return preflight, fmt.Errorf("resource %q budget must be non-negative", plan.Name)
		}
		if !allowPartial && plan.Budget < plan.Ceiling {
			return preflight, fmt.Errorf(
				"resource %q budget %d cannot cover ceiling %d",
				plan.Name,
				plan.Budget,
				plan.Ceiling,
			)
		}
		if plan.UnitUSD != nil && (math.IsNaN(*plan.UnitUSD) || math.IsInf(*plan.UnitUSD, 0) || *plan.UnitUSD < 0) {
			return preflight, fmt.Errorf("resource %q unit price must be finite and non-negative", plan.Name)
		}
		if plan.Ceiling == 0 {
			continue
		}
		if plan.UnitUSD == nil {
			preflight.MissingPrices = append(preflight.MissingPrices, plan.Name)
			continue
		}
		preflight.EstimatedUSD += float64(plan.Ceiling) * *plan.UnitUSD
	}
	if len(preflight.MissingPrices) > 0 && !allowUnpriced {
		return preflight, fmt.Errorf(
			"pricing unavailable for %s",
			strings.Join(preflight.MissingPrices, ", "),
		)
	}
	if preflight.EstimatedUSD > maximumUSD {
		return preflight, fmt.Errorf(
			"estimated provider cost %.6f exceeds maximum USD %.6f",
			preflight.EstimatedUSD,
			maximumUSD,
		)
	}
	return preflight, nil
}

// NewResourceBudgets constructs one finite budget and composed limiter per
// resource. A nil or missing rate limiter applies only the finite budget.
func NewResourceBudgets(
	plans []ResourcePlan,
	rates map[string]Limiter,
) (map[string]*Budget, map[string]Limiter, error) {
	budgets := make(map[string]*Budget, len(plans))
	limiters := make(map[string]Limiter, len(plans))
	for index, plan := range plans {
		if strings.TrimSpace(plan.Name) == "" {
			return nil, nil, fmt.Errorf("resource plan %d: name is required", index)
		}
		if _, ok := budgets[plan.Name]; ok {
			return nil, nil, fmt.Errorf("resource plan %q is duplicated", plan.Name)
		}
		budget, err := NewBudget(plan.Budget)
		if err != nil {
			return nil, nil, fmt.Errorf("create resource %q budget: %w", plan.Name, err)
		}
		budgets[plan.Name] = budget
		limiters[plan.Name] = Chain(budget, rates[plan.Name])
	}
	for name := range rates {
		if _, ok := budgets[name]; !ok {
			return nil, nil, fmt.Errorf("rate limiter has no resource plan %q", name)
		}
	}
	return budgets, limiters, nil
}
