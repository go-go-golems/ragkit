package execution

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestValidateResourcePlansPricingBudgetsAndPartialRuns(t *testing.T) {
	t.Parallel()
	embeddingPrice, generationPrice := 0.01, 0.10
	plans := []ResourcePlan{
		{Name: "embedding", Ceiling: 10, Budget: 10, UnitUSD: &embeddingPrice},
		{Name: "generation", Ceiling: 2, Budget: 2, UnitUSD: &generationPrice},
		{Name: "unused", Ceiling: 0, Budget: 0},
	}
	preflight, err := ValidateResourcePlans(plans, 0.31, false, false)
	require.NoError(t, err)
	require.InDelta(t, 0.30, preflight.EstimatedUSD, 1e-12)
	require.Empty(t, preflight.MissingPrices)

	partial := append([]ResourcePlan(nil), plans...)
	partial[0].Budget = 1
	_, err = ValidateResourcePlans(partial, 1, false, false)
	require.ErrorContains(t, err, "cannot cover ceiling")
	_, err = ValidateResourcePlans(partial, 1, false, true)
	require.NoError(t, err)
	_, err = ValidateResourcePlans(plans, 0.29, false, false)
	require.ErrorContains(t, err, "exceeds maximum USD")
}

func TestValidateResourcePlansRequiresUnpricedAuthorization(t *testing.T) {
	t.Parallel()
	plans := []ResourcePlan{{Name: "generation", Ceiling: 1, Budget: 1}}
	preflight, err := ValidateResourcePlans(plans, 0, false, false)
	require.ErrorContains(t, err, "pricing unavailable")
	require.Equal(t, []string{"generation"}, preflight.MissingPrices)
	preflight, err = ValidateResourcePlans(plans, 0, true, false)
	require.NoError(t, err)
	require.Zero(t, preflight.EstimatedUSD)
}

func TestValidateResourcePlansRejectsInvalidPlans(t *testing.T) {
	t.Parallel()
	price := -1.0
	nan := math.NaN()
	infinity := math.Inf(1)
	tests := []struct {
		name  string
		plans []ResourcePlan
	}{
		{name: "missing name", plans: []ResourcePlan{{}}},
		{name: "duplicate", plans: []ResourcePlan{{Name: "x"}, {Name: "x"}}},
		{name: "negative ceiling", plans: []ResourcePlan{{Name: "x", Ceiling: -1}}},
		{name: "negative budget", plans: []ResourcePlan{{Name: "x", Budget: -1}}},
		{name: "negative price", plans: []ResourcePlan{{Name: "x", Ceiling: 1, Budget: 1, UnitUSD: &price}}},
		{name: "NaN price", plans: []ResourcePlan{{Name: "x", Ceiling: 1, Budget: 1, UnitUSD: &nan}}},
		{name: "infinite price", plans: []ResourcePlan{{Name: "x", Ceiling: 1, Budget: 1, UnitUSD: &infinity}}},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := ValidateResourcePlans(test.plans, 1, false, false)
			require.Error(t, err)
		})
	}
	_, err := ValidateResourcePlans(nil, -1, false, false)
	require.ErrorContains(t, err, "maximum USD")
	_, err = ValidateResourcePlans(nil, math.NaN(), false, false)
	require.ErrorContains(t, err, "maximum USD")
}

func TestNewResourceBudgetsComposesRatesAndRejectsUnknownRates(t *testing.T) {
	t.Parallel()
	rate := &countingLimiter{}
	plans := []ResourcePlan{
		{Name: "embedding", Budget: 2},
		{Name: "generation", Budget: 1},
	}
	budgets, limiters, err := NewResourceBudgets(
		plans,
		map[string]Limiter{"embedding": rate},
	)
	require.NoError(t, err)
	require.NoError(t, limiters["embedding"].Wait(t.Context(), 2))
	require.Equal(t, 2, budgets["embedding"].Spent())
	require.Equal(t, 2, rate.units)
	require.NoError(t, limiters["generation"].Wait(t.Context(), 1))
	require.ErrorIs(t, limiters["generation"].Wait(t.Context(), 1), ErrBudgetExceeded)

	_, _, err = NewResourceBudgets(plans, map[string]Limiter{"unknown": rate})
	require.ErrorContains(t, err, "no resource plan")
}

type countingLimiter struct {
	units int
}

func (limiter *countingLimiter) Wait(_ context.Context, units int) error {
	if units < 1 {
		return errors.New("invalid units")
	}
	limiter.units += units
	return nil
}
