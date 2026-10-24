// Package httpheaders validates per-invocation HTTP header overrides.
package httpheaders

import (
	"errors"
	"net/http"
	"strings"
)

// Validate rejects names that differ only in case. Such entries would otherwise
// overwrite one another in Go map iteration order. Error messages omit values.
func Validate(headers http.Header) error {
	seen := make(map[string]struct{}, len(headers))
	for name := range headers {
		key := strings.ToLower(name)
		if _, duplicate := seen[key]; duplicate {
			return errors.New("header names must be unique ignoring case")
		}
		seen[key] = struct{}{}
	}
	return nil
}
