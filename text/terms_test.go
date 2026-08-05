package text

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTermsPreservesCurrentAnalysisPolicy(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		input string
		want  []string
	}{
		{name: "empty", input: "", want: []string{}},
		{name: "punctuation and repeats", input: "Oak, OAK! tree-42 oak", want: []string{"oak", "oak", "tree", "42", "oak"}},
		{name: "Unicode letters and numbers", input: "Érable 東京 １２", want: []string{"érable", "東京", "１２"}},
		{name: "symbols are separators", input: "oak🌳maple+pine", want: []string{"oak", "maple", "pine"}},
		{name: "combining marks are separators", input: "e\u0301cole", want: []string{"e", "cole"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, test.want, Terms(test.input))
		})
	}
}

func TestTermSetDeduplicates(t *testing.T) {
	t.Parallel()
	require.Equal(t, map[string]struct{}{"oak": {}, "tree": {}}, TermSet("Oak oak TREE"))
}
