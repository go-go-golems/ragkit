package digest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"hash"
	"io"
	"os"

	"github.com/pkg/errors"
)

// File returns the SHA-256 digest of a file while honoring cancellation.
func File(ctx context.Context, path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", errors.Wrap(err, "open digest input")
	}
	defer func() { _ = file.Close() }()
	hash := sha256.New()
	buffer := make([]byte, 64*1024)
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		read, readErr := file.Read(buffer)
		if read > 0 {
			_, _ = hash.Write(buffer[:read])
		}
		if readErr == io.EOF {
			return hex.EncodeToString(hash.Sum(nil)), nil
		}
		if readErr != nil {
			return "", errors.Wrap(readErr, "read digest input")
		}
	}
}

// Bytes returns the lowercase hexadecimal SHA-256 digest of data.
func Bytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// Text returns the SHA-256 digest of the exact UTF-8 bytes in value.
func Text(value string) string {
	return Bytes([]byte(value))
}

// JSON returns the digest of encoding/json's deterministic representation.
func JSON(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", errors.Wrap(err, "marshal digest input")
	}
	return Bytes(data), nil
}

// JSONSequence returns the same digest as JSON over a non-nil slice while
// retaining only the current value. The producer must yield values in their
// canonical order and must not call yield after returning. This is the small
// fold used by disk-backed builders whose complete input must not coexist in
// memory.
func JSONSequence[T any](ctx context.Context, produce func(yield func(T) error) error) (string, error) {
	if produce == nil {
		return "", errors.New("JSON sequence producer is required")
	}
	digester := &jsonSequenceDigester{hash: sha256.New()}
	_, _ = digester.hash.Write([]byte{'['})
	err := produce(func(value T) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		return digester.add(value)
	})
	if err != nil {
		return "", errors.Wrap(err, "produce JSON digest sequence")
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	_, _ = digester.hash.Write([]byte{']'})
	return hex.EncodeToString(digester.hash.Sum(nil)), nil
}

type jsonSequenceDigester struct {
	hash  hash.Hash
	count int
}

func (d *jsonSequenceDigester) add(value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return errors.Wrap(err, "marshal JSON digest sequence value")
	}
	if d.count > 0 {
		_, _ = d.hash.Write([]byte{','})
	}
	_, _ = d.hash.Write(data)
	d.count++
	return nil
}

// TruncatedJSON returns prefix followed by the first byteCount digest bytes
// encoded as lowercase hexadecimal.
func TruncatedJSON(prefix string, byteCount int, value any) (string, error) {
	if byteCount < 1 || byteCount > sha256.Size {
		return "", errors.Errorf(
			"digest truncation byte count %d is outside [1,%d]",
			byteCount,
			sha256.Size,
		)
	}
	data, err := json.Marshal(value)
	if err != nil {
		return "", errors.Wrap(err, "marshal digest input")
	}
	sum := sha256.Sum256(data)
	return prefix + hex.EncodeToString(sum[:byteCount]), nil
}
