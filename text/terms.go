package text

import (
	"strings"
	"unicode"
)

// Terms lowercases value and splits it wherever a rune is neither a Unicode
// letter nor a Unicode number. Terms preserves occurrence order and duplicates.
func Terms(value string) []string {
	return strings.FieldsFunc(strings.ToLower(value), func(char rune) bool {
		return !unicode.IsLetter(char) && !unicode.IsNumber(char)
	})
}

// TermSet returns the distinct terms produced by Terms.
func TermSet(value string) map[string]struct{} {
	terms := Terms(value)
	result := make(map[string]struct{}, len(terms))
	for _, term := range terms {
		result[term] = struct{}{}
	}
	return result
}
