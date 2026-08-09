package digest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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
