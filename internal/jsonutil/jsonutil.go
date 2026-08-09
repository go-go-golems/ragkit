package jsonutil

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"
)

// DecodeStrict decodes exactly one JSON value into T. Unknown object fields,
// malformed input, trailing JSON values, and trailing non-whitespace content
// are rejected.
func DecodeStrict[T any](data []byte) (T, error) {
	var value T
	if err := DecodeStrictInto(data, &value); err != nil {
		return value, err
	}
	return value, nil
}

// DecodeStrictInto decodes exactly one JSON value into target.
func DecodeStrictInto(data []byte, target any) error {
	if target == nil {
		return fmt.Errorf("JSON decode target is required")
	}
	value := reflect.ValueOf(target)
	if value.Kind() == reflect.Pointer && value.IsNil() {
		return fmt.Errorf("JSON decode target is nil")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("decode JSON value: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("JSON contains a trailing value")
		}
		return fmt.Errorf("decode trailing JSON content: %w", err)
	}
	return nil
}

// StripCompleteFence trims surrounding whitespace and removes one complete
// bare or json Markdown fence. Incomplete, nested, and other-language fences
// are returned unchanged after surrounding whitespace is trimmed.
func StripCompleteFence(value string) string {
	trimmed := strings.TrimSpace(value)
	firstNewline := strings.IndexByte(trimmed, '\n')
	if firstNewline < 0 {
		return trimmed
	}
	opener := strings.TrimSpace(trimmed[:firstNewline])
	if opener != "```" && opener != "```json" {
		return trimmed
	}
	bodyAndClose := trimmed[firstNewline+1:]
	lastNewline := strings.LastIndexByte(bodyAndClose, '\n')
	if lastNewline < 0 || strings.TrimSpace(bodyAndClose[lastNewline+1:]) != "```" {
		return trimmed
	}
	body := bodyAndClose[:lastNewline]
	if strings.Contains(body, "```") {
		return trimmed
	}
	return strings.TrimSpace(body)
}
