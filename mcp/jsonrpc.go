package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

	gonanoid "github.com/matoous/go-nanoid/v2"
)

// Codec handles encoding and decoding of JSON-RPC messages
type Codec struct {
	mu      sync.Mutex
	encoder *json.Encoder
	decoder *bufio.Scanner
}

// NewCodec creates a new JSON-RPC codec
func NewCodec(reader io.Reader, writer io.Writer) *Codec {
	scanner := bufio.NewScanner(reader)
	// Set a larger buffer for the scanner to handle large messages
	const maxScanTokenSize = 10 * 1024 * 1024 // 10MB
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, maxScanTokenSize)

	return &Codec{
		encoder: json.NewEncoder(writer),
		decoder: scanner,
	}
}

// WriteMessage writes a JSON-RPC message
func (c *Codec) WriteMessage(msg RpcMessage) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.encoder.Encode(msg)
}

// WriteBatch writes a batch of JSON-RPC messages
func (c *Codec) WriteBatch(messages []RpcMessage) error {
	if len(messages) == 0 {
		return fmt.Errorf("empty batch")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Convert to raw messages for batch encoding
	var batch []interface{}
	for _, msg := range messages {
		batch = append(batch, msg)
	}

	return c.encoder.Encode(batch)
}

// ReadMessage reads a JSON-RPC message
func (c *Codec) ReadMessage() ([]RpcMessage, error) {
	if !c.decoder.Scan() {
		if err := c.decoder.Err(); err != nil {
			return nil, err
		}
		return nil, io.EOF
	}

	line := c.decoder.Bytes()

	// Try to parse as a single message first
	var msg RpcMessage
	if err := json.Unmarshal(line, &msg); err == nil {
		return []RpcMessage{msg}, nil
	}

	// Try to parse as a batch
	var batch []json.RawMessage
	if err := json.Unmarshal(line, &batch); err != nil {
		// Not a batch either, return the original parse error
		return nil, fmt.Errorf("failed to parse message: %w", err)
	}

	// Parse each message in the batch
	var messages []RpcMessage
	for i, raw := range batch {
		var msg RpcMessage
		if err := json.Unmarshal(raw, &msg); err != nil {
			return nil, fmt.Errorf("failed to parse batch message %d: %w", i, err)
		}
		messages = append(messages, msg)
	}

	return messages, nil
}

// IDGenerator generates unique request IDs using nanoid
type IDGenerator struct {
	// No mutex needed since nanoid.New() is thread-safe
}

// NewIDGenerator creates a new request ID generator
func NewIDGenerator() *IDGenerator {
	return &IDGenerator{}
}

// Generate generates a new unique request ID using nanoid
func (g *IDGenerator) Generate() RequestID {
	// Generate a nanoid with default alphabet and length (21 characters)
	// This provides excellent uniqueness guarantees and is URL-safe
	id, err := gonanoid.New()
	if err != nil {
		// Fallback to a simple UUID-like string if nanoid fails
		// This should never happen in practice
		return fmt.Sprintf("fallback-%d", time.Now().UnixNano())
	}
	return id
}
