package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// StreamableHTTP implements Transport using Streamable HTTP (protocol version 2025-03-26).
type StreamableHTTP struct {
	config    HTTPConfig
	client    *http.Client
	sessionID string

	// Stream ID counter for unique stream identification
	streamCounter atomic.Uint64

	// State that needs synchronization
	connectedState atomic.Bool

	// Use sync.Once to ensure Connect is only called once
	connectOnce sync.Once

	// SSE stream management using sync.Map for concurrent access
	sseStreams sync.Map // map[string]*streamableSSEStream

	// Event ID needs synchronization as it's accessed from multiple SSE goroutines
	eventState struct {
		sync.Mutex
		nextEventID string
	}

	// Channel for receiving messages
	receiveChan chan MessageOrError

	// Use sync.Once to ensure Close is only called once
	closeOnce sync.Once
}

// streamableSSEStream represents an active SSE connection
type streamableSSEStream struct {
	response *http.Response
	reader   *bufio.Reader
	done     chan struct{}
}

// NewStreamableHTTP creates a new Streamable HTTP transport (protocol version 2025-03-26)
func NewStreamableHTTP(config HTTPConfig) *StreamableHTTP {
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{
			Timeout: 30 * time.Second,
		}
	}

	return &StreamableHTTP{
		config:      config,
		client:      client,
		receiveChan: make(chan MessageOrError, 1),
	}
}

// Connect establishes the HTTP transport connection.
func (t *StreamableHTTP) Connect(ctx context.Context) error {
	var result error
	t.connectOnce.Do(func() {
		if t.connectedState.Load() {
			result = ErrAlreadyConnected
			return
		}
		t.connectedState.Store(true)
	})
	return result
}

// Close closes the HTTP transport connection.
// This method is thread-safe and can be called concurrently multiple times.
func (t *StreamableHTTP) Close() error {
	var err error
	t.closeOnce.Do(func() {
		t.connectedState.Store(false)
		t.sseStreams.Range(func(key, value interface{}) bool {
			if stream, ok := value.(*streamableSSEStream); ok {
				close(stream.done)
				errClose := stream.response.Body.Close()
				if errClose != nil && err == nil {
					err = errClose
				}
			}
			t.sseStreams.Delete(key)
			return true
		})
		close(t.receiveChan)
		if t.sessionID != "" {
			req, reqErr := http.NewRequest(http.MethodDelete, t.config.URL, nil)
			if reqErr == nil {
				t.addHeaders(req)
				req.Header.Set("Mcp-Session-Id", t.sessionID)
				t.client.Do(req)
			}
		}
	})
	return err
}

// Send sends a JSON-RPC message via HTTP POST
func (t *StreamableHTTP) Send(msg RpcMessage) error {
	if !t.connectedState.Load() {
		return ErrNotConnected
	}

	// Marshal single message
	body, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, t.config.URL, bytes.NewReader(body))
	if err != nil {
		return NewTransportError("streamable-http", "create request", err)
	}

	t.addHeaders(req)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")

	resp, err := t.client.Do(req)
	if err != nil {
		return NewTransportError("streamable-http", "send request", err)
	}

	// Check status code
	if resp.StatusCode == http.StatusAccepted {
		// Accepted, no response body
		resp.Body.Close()
		return nil
	}

	if resp.StatusCode == http.StatusUnauthorized {
		// Authentication error
		resp.Body.Close()
		return AuthenticationError
	}

	if resp.StatusCode == http.StatusForbidden {
		// Forbidden - insufficient permissions
		resp.Body.Close()
		return fmt.Errorf("forbidden: insufficient permissions (HTTP 403)")
	}

	if resp.StatusCode == http.StatusTooManyRequests {
		// Rate limit error
		retryAfter := 0
		if ra := resp.Header.Get("Retry-After"); ra != "" {
			retryAfter, _ = strconv.Atoi(ra)
		}
		resp.Body.Close()
		return NewRateLimitError(retryAfter, "HTTP 429 Too Many Requests")
	}

	if resp.StatusCode >= 400 {
		// Check if this is an initialize request and we got 404/405
		// This indicates an old server that doesn't support the new protocol
		if msg.Method == "initialize" &&
			(resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMethodNotAllowed) {
			resp.Body.Close()
			return ErrLegacyHTTPSSERequired
		}

		// Error response
		defer resp.Body.Close()
		var errResp RpcMessage
		b, _ := io.ReadAll(resp.Body)
		if err = json.Unmarshal(b, &errResp); err != nil {
			// Return a general error with the status code if we can't parse the response
			return NewTransportError("streamable-http", "decode error response",
				fmt.Errorf("HTTP %d: %s", resp.StatusCode, http.StatusText(resp.StatusCode)))
		}
		if errResp.Error != nil {
			return NewProtocolError(errResp.Error.Code, errResp.Error.Message, errResp.Error.Data)
		}
		// Return a general error with the status code if no error details in response
		return NewTransportError("streamable-http", "request failed",
			fmt.Errorf("HTTP %d: %s", resp.StatusCode, http.StatusText(resp.StatusCode)))
	}

	// Check if this is the initialize response
	if msg.Method == "initialize" && msg.ID != "" {
		// Extract session ID from response headers
		if sessionID := resp.Header.Get("Mcp-Session-Id"); sessionID != "" {
			t.sessionID = sessionID
		}
	}

	// Handle response based on content type
	contentType := resp.Header.Get("Content-Type")
	if strings.HasPrefix(contentType, "text/event-stream") {
		// SSE response, set up stream
		return t.handleSSEResponse(resp)
	} else {
		// Regular JSON response - send immediately via receive channel
		defer resp.Body.Close()

		// Parse response
		var response interface{}
		if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
			return NewTransportError("streamable-http", "decode response", err)
		}

		// Convert to message
		data, err := json.Marshal(response)
		if err != nil {
			return err
		}
		var responseMsg RpcMessage
		if err := json.Unmarshal(data, &responseMsg); err != nil {
			return err
		}

		// Send message immediately via receive channel
		select {
		case t.receiveChan <- MessageOrError{Message: responseMsg}:
		default:
			// Channel is full, this shouldn't happen with buffer size 1 and immediate consumption
			// but we handle it gracefully
		}

		return nil
	}
}

// handleSSEResponse handles a Server-Sent Events response
func (t *StreamableHTTP) handleSSEResponse(resp *http.Response) error {
	stream := &streamableSSEStream{
		response: resp,
		reader:   bufio.NewReader(resp.Body),
		done:     make(chan struct{}),
	}

	// Store stream using sync.Map with atomic counter for unique ID
	streamID := fmt.Sprintf("stream-%d", t.streamCounter.Add(1))
	t.sseStreams.Store(streamID, stream)

	// Start reading SSE events in a separate goroutine
	go t.readSSEStream(streamID, stream)

	return nil
}

// readSSEStream reads events from an SSE stream and sends them directly to the receive channel
func (t *StreamableHTTP) readSSEStream(streamID string, stream *streamableSSEStream) {
	defer func() {
		stream.response.Body.Close()
		t.sseStreams.Delete(streamID)
	}()

	var eventData bytes.Buffer
	var eventID string

	for {
		select {
		case <-stream.done:
			return
		default:
		}

		line, err := stream.reader.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				// Send error to receive channel
				select {
				case t.receiveChan <- MessageOrError{Error: err}:
				case <-stream.done:
				}
			}
			return
		}

		line = strings.TrimSpace(line)

		if line == "" {
			// End of event
			if eventData.Len() > 0 {
				// Parse the event data as a JSON-RPC message
				var msg RpcMessage
				if err := json.Unmarshal(eventData.Bytes(), &msg); err != nil {
					// Send parse error to receive channel
					select {
					case t.receiveChan <- MessageOrError{Error: fmt.Errorf("failed to parse SSE event: %w", err)}:
					case <-stream.done:
						return
					}
				} else {
					// Send message directly to receive channel
					select {
					case t.receiveChan <- MessageOrError{Message: msg}:
					case <-stream.done:
						return
					}
				}

				// Update last event ID
				if eventID != "" {
					t.eventState.Lock()
					t.eventState.nextEventID = eventID
					t.eventState.Unlock()
				}

				eventData.Reset()
				eventID = ""
			}
			continue
		}

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			eventData.WriteString(data)
		} else if strings.HasPrefix(line, "id: ") {
			eventID = strings.TrimPrefix(line, "id: ")
		}
		// Ignore other fields like event:, retry:, etc.
	}
}

// Receive returns a channel that delivers messages or errors.
// The channel will be closed when the transport is closed.
func (t *StreamableHTTP) Receive() <-chan MessageOrError {
	return t.receiveChan
}

// openSSEStream opens a new SSE stream via GET request
func (t *StreamableHTTP) openSSEStream() error {
	req, err := http.NewRequest(http.MethodGet, t.config.URL, nil)
	if err != nil {
		return NewTransportError("streamable-http", "create GET request", err)
	}

	t.addHeaders(req)
	req.Header.Set("Accept", "text/event-stream")

	// Add Last-Event-ID if we have one
	t.eventState.Lock()
	eventID := t.eventState.nextEventID
	t.eventState.Unlock()

	if eventID != "" {
		req.Header.Set("Last-Event-ID", eventID)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return NewTransportError("streamable-http", "send GET request", err)
	}

	if resp.StatusCode == 405 {
		// Method not allowed, server doesn't support SSE via GET
		resp.Body.Close()
		return nil
	}

	if resp.StatusCode >= 400 {
		resp.Body.Close()
		return NewTransportError("streamable-http", "GET request failed",
			fmt.Errorf("HTTP %d", resp.StatusCode))
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "text/event-stream") {
		resp.Body.Close()
		return NewTransportError("streamable-http", "unexpected content type",
			fmt.Errorf("expected text/event-stream, got %s", contentType))
	}

	return t.handleSSEResponse(resp)
}

// addHeaders adds common headers to a request
func (t *StreamableHTTP) addHeaders(req *http.Request) {
	// Add custom headers
	for k, v := range t.config.Headers {
		req.Header.Set(k, v)
	}

	// Add session ID if we have one
	if t.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", t.sessionID)
	}

	// Add protocol version for new transport
	req.Header.Set("MCP-Protocol-Version", ProtocolVersion)
}
