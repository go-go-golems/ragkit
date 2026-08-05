package vector

import (
	"errors"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCosine(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		left, right []float32
		want        float64
	}{
		{name: "identical", left: []float32{1, 2}, right: []float32{1, 2}, want: 1},
		{name: "orthogonal", left: []float32{1, 0}, right: []float32{0, 1}, want: 0},
		{name: "opposite", left: []float32{1, 0}, right: []float32{-1, 0}, want: -1},
		{name: "zero", left: []float32{0, 0}, right: []float32{1, 0}, want: 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			left, right := append([]float32(nil), test.left...), append([]float32(nil), test.right...)
			got, err := Cosine(left, right)
			require.NoError(t, err)
			require.InDelta(t, test.want, got, 1e-12)
			require.Equal(t, test.left, left)
			require.Equal(t, test.right, right)
		})
	}
}

func TestVectorValidation(t *testing.T) {
	t.Parallel()
	_, err := Cosine([]float32{1}, []float32{1, 2})
	require.ErrorIs(t, err, ErrDimensionMismatch)
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		require.ErrorIs(t, ValidateFinite([]float32{value}), ErrNonFinite)
	}
	require.True(t, errors.Is(ValidateDimensions(2, []float32{1}), ErrDimensionMismatch))
}

func TestNormalize(t *testing.T) {
	t.Parallel()
	values := []float32{3, 4}
	require.NoError(t, Normalize(values))
	require.InDelta(t, 0.6, values[0], 1e-6)
	require.InDelta(t, 0.8, values[1], 1e-6)
	zero := []float32{0, 0}
	require.NoError(t, Normalize(zero))
	require.Equal(t, []float32{0, 0}, zero)
}
