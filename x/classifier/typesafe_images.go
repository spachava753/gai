package classifier

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"strings"

	"github.com/spachava753/gai"
	wire "github.com/spachava753/gai/internal/typesafe"
)

// typeSafeImages sends inline images, without changing the caller's values.
// Explicit IDs permit references anywhere guidance accepts a JSON string.
func typeSafeImages(body *wire.ClassificationRequest, media []gai.Block) error {
	if len(media) == 0 {
		return nil
	}
	assets := make(map[string]wire.ImageAsset, len(media))
	var prefix []json.RawMessage
	original, err := json.Marshal(body)
	if err != nil {
		return err
	}
	totalKeyBytes := 0
	for _, block := range media {
		if block.ModalityType != gai.Image || (block.MimeType != "image/png" && block.MimeType != "image/jpeg") {
			return gai.UnsupportedInputModalityErr(block.MimeType)
		}
		id := block.ID
		if id == "" {
			for {
				id = "gai-image-" + rand.Text()
				if _, duplicate := assets[id]; !duplicate && !bytes.Contains(original, []byte(id)) {
					break
				}
			}
			encoded, err := json.Marshal(id)
			if err != nil {
				return err
			}
			prefix = append(prefix, encoded)
		} else {
			if strings.TrimSpace(id) == "" || len(id) > 1024 {
				return invalid("Input.Media.ID", "asset references require 1 to 1024 UTF-8 bytes")
			}
			if !typeSafeReference(body.State, id) {
				found := false
				for _, question := range body.Questions {
					// Only instructions and criteria are reference-bearing, not type.
					var q wire.Question
					if err := json.Unmarshal(question, &q); err != nil {
						return err
					}
					if typeSafeReference(optionalRaw(q.Instructions), id) || typeSafeReference(optionalRaw(q.Criteria), id) {
						found = true
						break
					}
				}
				if !found {
					return invalid("Input.Media.ID", "every named image must be referenced in data, instructions, or criteria")
				}
			}
		}
		if _, duplicate := assets[id]; duplicate {
			return invalid("Input.Media.ID", "duplicate asset reference")
		}
		totalKeyBytes += len(id)
		if totalKeyBytes > 65536 {
			return invalid("Input.Media.ID", "asset references exceed 65536 bytes")
		}
		format := "png"
		if block.MimeType == "image/jpeg" {
			format = "jpeg"
		}
		assets[id] = wire.ImageAsset{Type: "image", Format: format, DataBase64: block.Content.String()}
	}
	if len(prefix) > 0 {
		prefix = append(prefix, body.State)
		body.State, err = json.Marshal(prefix)
		if err != nil {
			return err
		}
	}
	body.Assets = &assets
	return nil
}

func optionalRaw(value *json.RawMessage) json.RawMessage {
	if value == nil {
		return nil
	}
	return *value
}

// typeSafeReference matches string values, never object keys or number spellings.
func typeSafeReference(data json.RawMessage, reference string) bool {
	if len(data) == 0 {
		return false
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if decoder.Decode(&value) != nil {
		return false
	}
	var contains func(any) bool
	contains = func(value any) bool {
		switch value := value.(type) {
		case string:
			return strings.Contains(value, reference)
		case []any:
			for _, item := range value {
				if contains(item) {
					return true
				}
			}
		case map[string]any:
			for _, item := range value {
				if contains(item) {
					return true
				}
			}
		}
		return false
	}
	return contains(value)
}
