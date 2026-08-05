package vector

import (
	"errors"
	"fmt"
	"math"
)

var (
	ErrDimensionMismatch = errors.New("vector dimensions differ")
	ErrNonFinite         = errors.New("vector contains a non-finite component")
)

func ValidateFinite(values []float32) error {
	for index, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("%w at component %d", ErrNonFinite, index)
		}
	}
	return nil
}

func ValidateDimensions(expected int, values []float32) error {
	if expected < 1 || len(values) != expected {
		return fmt.Errorf("%w: got %d, want %d", ErrDimensionMismatch, len(values), expected)
	}
	return nil
}

// Cosine returns cosine similarity using float64 accumulation. A zero vector
// has similarity zero with every vector. Inputs are not modified.
func Cosine(left, right []float32) (float64, error) {
	if err := ValidateDimensions(len(left), right); err != nil {
		return 0, err
	}
	if err := ValidateFinite(left); err != nil {
		return 0, err
	}
	if err := ValidateFinite(right); err != nil {
		return 0, err
	}
	var dot, leftNorm, rightNorm float64
	for index := range left {
		l, r := float64(left[index]), float64(right[index])
		dot += l * r
		leftNorm += l * l
		rightNorm += r * r
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 0, nil
	}
	return dot / math.Sqrt(leftNorm*rightNorm), nil
}

// Normalize scales values in place to unit length using float64 accumulation.
// A zero vector is preserved.
func Normalize(values []float32) error {
	if err := ValidateFinite(values); err != nil {
		return err
	}
	var squaredNorm float64
	for _, value := range values {
		squaredNorm += float64(value) * float64(value)
	}
	if squaredNorm == 0 {
		return nil
	}
	norm := float32(math.Sqrt(squaredNorm))
	for index := range values {
		values[index] /= norm
	}
	return nil
}
