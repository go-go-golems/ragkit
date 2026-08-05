package jsonutil

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

type fixture struct {
	Value string `json:"value"`
}

func TestDecodeStrict(t *testing.T) {
	t.Parallel()
	value, err := DecodeStrict[fixture]([]byte("{\"value\":\"ok\"} \n"))
	require.NoError(t, err)
	require.Equal(t, fixture{Value: "ok"}, value)
	for _, input := range []string{
		`{"value":"ok","unknown":true}`,
		`{"value":`,
		`{"value":"ok"} {"value":"second"}`,
		`{"value":"ok"} trailing`,
	} {
		_, err := DecodeStrict[fixture]([]byte(input))
		require.Error(t, err)
	}
	var target *fixture
	require.Error(t, DecodeStrictInto([]byte(`{}`), target))
	require.Error(t, DecodeStrictInto([]byte(`{}`), nil))
}

func TestStripCompleteFence(t *testing.T) {
	t.Parallel()
	require.Equal(t, `{"value":"ok"}`, StripCompleteFence("```\n{\"value\":\"ok\"}\n```"))
	require.Equal(t, `{"value":"ok"}`, StripCompleteFence("```json\n{\"value\":\"ok\"}\n```"))
	for _, input := range []string{
		"```json\n{}",
		"```go\n{}\n```",
		"```json\n```\n{}\n```",
		"```json {} ```",
	} {
		require.Equal(t, strings.TrimSpace(input), StripCompleteFence(input))
	}
}
