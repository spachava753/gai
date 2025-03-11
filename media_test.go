package gai

import (
	"encoding/base64"
	"encoding/json"
	"github.com/google/go-cmp/cmp"
	"testing"
)

func TestMedia_MarshalJSON(t *testing.T) {
	// Create a test Media object
	testBody := []byte("test binary data")
	media := Media{
		Mimetype: "application/octet-stream",
		Body:     testBody,
	}

	// Marshal to JSON
	jsonData, err := json.Marshal(&media)
	if err != nil {
		t.Fatalf("Failed to marshal Media: %v", err)
	}

	// Unmarshal to map for comparison (to avoid field order issues)
	var actualMap map[string]interface{}
	if err := json.Unmarshal(jsonData, &actualMap); err != nil {
		t.Fatalf("Failed to unmarshal actual JSON: %v", err)
	}

	// Create expected map
	expectedMap := map[string]interface{}{
		"mimetype": "application/octet-stream",
		"body":     base64.StdEncoding.EncodeToString(testBody),
	}

	// Compare actual and expected maps
	if diff := cmp.Diff(actualMap, expectedMap); diff != "" {
		t.Errorf("JSON content mismatch\ndiff:\n%s", diff)
	}
}

func TestMedia_UnmarshalJSON(t *testing.T) {
	// Create test data
	testBody := []byte("test binary data")
	encodedBody := base64.StdEncoding.EncodeToString(testBody)

	// Create JSON with base64 encoded body
	jsonData, err := json.Marshal(map[string]interface{}{
		"mimetype": "application/octet-stream",
		"body":     encodedBody,
	})
	if err != nil {
		t.Fatalf("Failed to create test JSON: %v", err)
	}

	// Unmarshal JSON to Media
	var media Media
	if err := json.Unmarshal(jsonData, &media); err != nil {
		t.Fatalf("Failed to unmarshal Media: %v", err)
	}

	// Verify fields
	if media.Mimetype != "application/octet-stream" {
		t.Errorf("Mimetype mismatch\nGot:      %s\nExpected: %s", media.Mimetype, "application/octet-stream")
	}

	if diff := cmp.Diff(media.Body, testBody); diff != "" {
		t.Errorf("Body mismatch\ndiff:\n%s", diff)
	}
}

func TestMedia_RoundTrip(t *testing.T) {
	// Create original Media
	original := Media{
		Mimetype: "image/jpeg",
		Body:     []byte("fake image data"),
	}

	// Marshal to JSON
	jsonData, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("Failed to marshal Media: %v", err)
	}

	// Unmarshal back to Media
	var decoded Media
	if err := json.Unmarshal(jsonData, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal Media: %v", err)
	}

	// Compare original and decoded
	if decoded.Mimetype != original.Mimetype {
		t.Errorf("Mimetype mismatch after round trip\nGot:      %s\nExpected: %s",
			decoded.Mimetype, original.Mimetype)
	}

	if diff := cmp.Diff(decoded.Body, original.Body); diff != "" {
		t.Errorf("Body mismatch after round trip\ndiff:\n%s", diff)
	}
}
