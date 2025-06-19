package mcp

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"time"
)

// HTTPConfig contains configuration for HTTP transport
type HTTPConfig struct {
	// Timeout is the default timeout for operations
	Timeout int `json:"timeout,omitempty"`

	// URL is the MCP endpoint URL
	URL string `json:"url"`

	// Headers to include in requests
	Headers map[string]string `json:"headers,omitempty"`

	// HTTPClient allows using a custom HTTP client
	HTTPClient *http.Client `json:"-"`

	// AllowedOrigins specifies which origins are allowed for CORS (for server implementations)
	// If empty, only same-origin requests are allowed
	AllowedOrigins []string `json:"allowed_origins,omitempty"`
}

// HTTP implements Transport with automatic protocol detection.
// It first tries the new Streamable HTTP transport (protocol version 2025-03-26),
// and falls back to HTTP+SSE (protocol version 2024-11-05) if the server doesn't support it.
//
// Thread Safety: This transport has mixed concurrency requirements:
// - Connect/Close must not be called concurrently
// - Send/Receive are synchronized internally by the underlying transport
type HTTP struct {
	config HTTPConfig

	// Synchronize access to transport and connected state
	mu        sync.RWMutex
	transport Transport // Either StreamableHTTP or HTTPSSE
	connected bool
}

// NewHTTP creates a new HTTP transport with automatic protocol detection
func NewHTTP(config HTTPConfig) *HTTP {
	if config.HTTPClient == nil {
		config.HTTPClient = &http.Client{
			Timeout: 30 * time.Second,
		}
	}

	return &HTTP{
		config: config,
	}
}

// Connect establishes the HTTP transport connection.
// It automatically detects whether to use Streamable HTTP or HTTP+SSE.
// Not thread-safe - must not be called concurrently with other methods.
func (t *HTTP) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.connected {
		return ErrAlreadyConnected
	}

	// First try Streamable HTTP (new protocol)
	streamable := NewStreamableHTTP(t.config)
	err := streamable.Connect(ctx)
	if err == nil {
		t.transport = streamable
		t.connected = true
		return nil
	}

	// If not ErrLegacyHTTPSSERequired, return the original error
	if !errors.Is(err, ErrLegacyHTTPSSERequired) {
		return err
	}

	// Fall back to HTTP+SSE (old protocol)
	httpSSE := NewHTTPSSE(t.config)
	err = httpSSE.Connect(ctx)
	if err != nil {
		return err
	}

	t.transport = httpSSE
	t.connected = true
	return nil
}

// Close closes the HTTP transport connection.
// Not thread-safe - must not be called concurrently with other methods.
func (t *HTTP) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.connected {
		return nil
	}

	if t.transport != nil {
		err := t.transport.Close()
		t.transport = nil
		t.connected = false
		return err
	}

	t.connected = false
	return nil
}

// Send sends a JSON-RPC message.
// Thread-safe if the underlying transport is thread-safe.
func (t *HTTP) Send(msg RpcMessage) error {
	t.mu.RLock()
	connected := t.connected
	transport := t.transport
	t.mu.RUnlock()

	if !connected || transport == nil {
		return ErrNotConnected
	}

	// Check if we need to detect protocol on first initialize
	if msg.Method == "initialize" && transport != nil {
		// Try to send with current transport
		err := transport.Send(msg)

		// If we get ErrLegacyHTTPSSERequired, switch transports
		if errors.Is(err, ErrLegacyHTTPSSERequired) {
			t.mu.Lock()
			// Double-check we still have the same transport
			if t.transport == transport {
				// Close current transport
				t.transport.Close()

				// Create and connect HTTP+SSE transport
				httpSSE := NewHTTPSSE(t.config)
				if connErr := httpSSE.Connect(context.Background()); connErr != nil {
					t.transport = nil
					t.connected = false
					t.mu.Unlock()
					return connErr
				}

				t.transport = httpSSE
				transport = httpSSE
			}
			t.mu.Unlock()

			// Retry the send
			return transport.Send(msg)
		}

		return err
	}

	return transport.Send(msg)
}

// Receive receives JSON-RPC messages.
// Thread-safe if the underlying transport is thread-safe.
func (t *HTTP) Receive() ([]RpcMessage, error) {
	t.mu.RLock()
	connected := t.connected
	transport := t.transport
	t.mu.RUnlock()

	if !connected || transport == nil {
		return nil, ErrNotConnected
	}

	return transport.Receive()
}
