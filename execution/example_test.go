package execution_test

import (
	"context"
	"fmt"
	"time"

	"github.com/go-go-golems/ragkit/execution"
)

func ExampleMap_budgetAndRate() {
	ctx := context.Background()

	budget, err := execution.NewBudget(6)
	if err != nil {
		panic(err)
	}
	rate, err := execution.NewTokenBucket(execution.Rate{
		Units: 100,
		Per:   time.Second,
		Burst: 3,
	})
	if err != nil {
		panic(err)
	}
	defer func() {
		if err := rate.Close(); err != nil {
			panic(err)
		}
	}()

	values, err := execution.Map(
		ctx,
		[]int{1, 2, 3},
		execution.MapOptions[int]{
			Workers: 3,
			Limiter: execution.Chain(budget, rate),
			Cost:    func(value int) int { return value },
		},
		func(_ context.Context, value int) (int, error) {
			return value * value, nil
		},
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(values)
	fmt.Println(budget.Snapshot())
	// Output:
	// [1 4 9]
	// {6 6 0}
}
