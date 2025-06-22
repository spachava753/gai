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

// ReadMessage reads a JSON-RPC message
func (c *Codec) ReadMessage() (RpcMessage, error) {
	if !c.decoder.Scan() {
		if err := c.decoder.Err(); err != nil {
			return RpcMessage{}, err
		}
		return RpcMessage{}, io.EOF
	}

	line := c.decoder.Bytes()

	// Parse as a single message
	var msg RpcMessage
	if err := json.Unmarshal(line, &msg); err != nil {
		return RpcMessage{}, fmt.Errorf("failed to parse message: %w", err)
	}

	return msg, nil
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
