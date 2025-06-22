package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
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

// HTTPSSE implements Transport using HTTP with Server-Sent Events (old protocol version 2024-11-05).
type HTTPSSE struct {
	config HTTPConfig
	client *http.Client

	connectedState atomic.Bool
	postEndpoint   atomic.Value // string
	sseStream      atomic.Value // httpSSEStream Single active SSE stream

	endpointChan     chan struct{}
	endpointChanOnce sync.Once

	// Channel for receiving messages
	receiveChan chan MessageOrError
	receiveOnce sync.Once
}

// httpSSEStream represents the active SSE connection
type httpSSEStream struct {
	response *http.Response
	reader   *bufio.Reader
	messages chan RpcMessage
	errors   chan error
	done     chan struct{}
}

// NewHTTPSSE creates a new HTTP+SSE transport (old protocol version 2024-11-05)
func NewHTTPSSE(config HTTPConfig) *HTTPSSE {
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{
			Timeout: 30 * time.Second,
		}
	}

	return &HTTPSSE{
		config:       config,
		client:       client,
		endpointChan: make(chan struct{}),
	}
}

// Connect establishes the HTTP+SSE transport connection.
func (t *HTTPSSE) Connect(ctx context.Context) error {
	if t.connectedState.Load() {
		return ErrAlreadyConnected
	}

	// Issue a GET request to establish SSE connection and get endpoint
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, t.config.URL, nil)
	if err != nil {
		return NewTransportError("http+sse", "create GET request", err)
	}

	t.addHeaders(req)
	req.Header.Set("Accept", "text/event-stream")

	resp, err := t.client.Do(req)
	if err != nil {
		return NewTransportError("http+sse", "send GET request", err)
	}

	if resp.StatusCode >= 400 {
		resp.Body.Close()
		if resp.StatusCode == http.StatusUnauthorized {
			return AuthenticationError
		}
		return NewTransportError("http+sse", "GET request failed",
			fmt.Errorf("HTTP %d", resp.StatusCode))
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "text/event-stream") {
		resp.Body.Close()
		return NewTransportError("http+sse", "unexpected content type",
			fmt.Errorf("expected text/event-stream, got %s", contentType))
	}

	// Create the SSE stream
	stream := httpSSEStream{
		response: resp,
		reader:   bufio.NewReader(resp.Body),
		messages: make(chan RpcMessage),
		errors:   make(chan error),
		done:     make(chan struct{}),
	}
	t.sseStream.Store(stream)

	// Start reading SSE events
	go t.readSSEStream()

	select {
	case <-ctx.Done():
		// Clean up on timeout
		close(stream.done)
		resp.Body.Close()
		return ctx.Err()
	case <-t.endpointChan:
		// endpoint found
	}

	t.connectedState.Store(true)
	return nil
}

// Close closes the HTTP+SSE transport connection.
func (t *HTTPSSE) Close() error {
	if !t.connectedState.Load() {
		return nil
	}

	t.connectedState.Store(false)

	// Close the SSE stream
	if t.sseStream.Load() != nil {
		sseStream := t.sseStream.Load().(httpSSEStream)
		close(sseStream.done)
		sseStream.response.Body.Close()
	}

	return nil
}

// Send sends a JSON-RPC message via HTTP POST
func (t *HTTPSSE) Send(msg RpcMessage) error {
	if !t.connectedState.Load() {
		return ErrNotConnected
	}

	endpointVal := t.postEndpoint.Load()
	if endpointVal == nil {
		return NewTransportError("http+sse", "send", fmt.Errorf("no POST endpoint available"))
	}
	endpoint := endpointVal.(string)

	if endpoint == "" {
		return NewTransportError("http+sse", "send", fmt.Errorf("no POST endpoint available"))
	}

	// Marshal single message
	body, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return NewTransportError("http+sse", "create request", err)
	}

	t.addHeaders(req)
	req.Header.Set("Content-Type", "application/json")

	resp, err := t.client.Do(req)
	if err != nil {
		return NewTransportError("http+sse", "send request", err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode == http.StatusUnauthorized {
		return AuthenticationError
	}

	if resp.StatusCode == http.StatusForbidden {
		return fmt.Errorf("forbidden: insufficient permissions (HTTP 403)")
	}

	if resp.StatusCode == http.StatusTooManyRequests {
		retryAfter := 0
		if ra := resp.Header.Get("Retry-After"); ra != "" {
			retryAfter, _ = strconv.Atoi(ra)
		}
		return NewRateLimitError(retryAfter, "HTTP 429 Too Many Requests")
	}

	if resp.StatusCode >= 400 {
		// Error response
		var errResp RpcMessage
		if err = json.NewDecoder(resp.Body).Decode(&errResp); err != nil {
			return NewTransportError("http+sse", "decode error response",
				fmt.Errorf("HTTP %d: %s", resp.StatusCode, http.StatusText(resp.StatusCode)))
		}
		if errResp.Error != nil {
			return NewProtocolError(errResp.Error.Code, errResp.Error.Message, errResp.Error.Data)
		}
		return NewTransportError("http+sse", "request failed",
			fmt.Errorf("HTTP %d: %s", resp.StatusCode, http.StatusText(resp.StatusCode)))
	}

	if !slices.Contains([]int{http.StatusOK, http.StatusAccepted}, resp.StatusCode) {
		return NewTransportError("http+sse", "request failed",
			fmt.Errorf(
				"unknown HTTP status code returned from server %d: %s",
				resp.StatusCode,
				http.StatusText(resp.StatusCode),
			))
	}

	return nil
}

// Receive returns a channel that delivers messages or errors.
// The channel will be closed when the transport is closed.
func (t *HTTPSSE) Receive() <-chan MessageOrError {
	// Ensure we only create the goroutine once
	t.receiveOnce.Do(func() {
		t.receiveChan = make(chan MessageOrError, 1)
		go t.receiveLoop()
	})
	return t.receiveChan
}

// receiveLoop continuously reads messages from the SSE stream
func (t *HTTPSSE) receiveLoop() {
	defer close(t.receiveChan)

	for {
		if !t.connectedState.Load() {
			return
		}

		if t.sseStream.Load() == nil {
			t.receiveChan <- MessageOrError{Error: NewTransportError("http+sse", "receive", fmt.Errorf("no SSE stream available"))}
			return
		}

		stream := t.sseStream.Load().(httpSSEStream)

		select {
		case msg := <-stream.messages:
			select {
			case t.receiveChan <- MessageOrError{Message: msg}:
			case <-stream.done:
				return
			}
		case err := <-stream.errors:
			t.receiveChan <- MessageOrError{Error: err}
			return
		case <-stream.done:
			return
		}
	}
}

// readSSEStream reads events from the SSE stream
func (t *HTTPSSE) readSSEStream() {
	stream := t.sseStream.Load().(httpSSEStream)
	defer func() {
		stream.response.Body.Close()
		close(stream.messages)
		close(stream.errors)
	}()

	var eventData bytes.Buffer
	var eventType string

	for {
		select {
		case <-stream.done:
			return
		default:
		}

		line, err := stream.reader.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				select {
				case stream.errors <- err:
				case <-stream.done:
				}
			}
			return
		}

		line = strings.TrimSpace(line)

		if line == "" {
			// End of event
			if eventData.Len() > 0 {
				// Check if this is an endpoint event
				if eventType == "endpoint" {
					// Extract the endpoint URL
					endpointData := strings.TrimSpace(eventData.String())

					// Resolve relative URLs against the base URL
					if !strings.HasPrefix(endpointData, "http://") && !strings.HasPrefix(endpointData, "https://") {
						// Parse base URL
						baseURL, err := url.Parse(t.config.URL)
						if err == nil {
							// Resolve endpoint URL against base
							endpointURL, err := baseURL.Parse(endpointData)
							if err == nil {
								endpointData = endpointURL.String()
							}
						}
					}

					t.postEndpoint.Store(endpointData)
					t.endpointChanOnce.Do(func() {
						close(t.endpointChan)
					})
				} else {
					// Parse the event data as a JSON-RPC message
					var msg RpcMessage
					if err := json.Unmarshal(eventData.Bytes(), &msg); err != nil {
						select {
						case stream.errors <- fmt.Errorf("failed to parse SSE event: %w", err):
						case <-stream.done:
						}
					} else {
						select {
						case stream.messages <- msg:
						case <-stream.done:
							return
						}
					}
				}

				eventData.Reset()
				eventType = ""
			}
			continue
		}

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			eventData.WriteString(data)
		} else if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
		}
		// Ignore other fields like id:, retry:, etc.
	}
}

// addHeaders adds common headers to a request
func (t *HTTPSSE) addHeaders(req *http.Request) {
	// Add custom headers
	for k, v := range t.config.Headers {
		req.Header.Set(k, v)
	}

	// No session ID or protocol version for old protocol
}
