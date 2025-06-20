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
// Thread Safety: This transport has mixed concurrency requirements:
// - Connect/Close must not be called concurrently
// - Send/Receive are synchronized internally due to HTTP requests and SSE streams
// - SSE stream management uses sync.Map for safe concurrent access
type StreamableHTTP struct {
	config    HTTPConfig
	client    *http.Client
	sessionID string

	// Stream ID counter for unique stream identification
	streamCounter atomic.Uint64

	// State that needs synchronization
	connectedState atomic.Bool

	// SSE stream management using sync.Map for concurrent access
	sseStreams sync.Map // map[string]*streamableSSEStream

	// Event ID needs synchronization as it's accessed from multiple SSE goroutines
	eventState struct {
		sync.Mutex
		nextEventID string
	}

	// Message store requires synchronization for cross-request state
	messageStore struct {
		sync.Mutex
		messages [][]RpcMessage
	}
}

// streamableSSEStream represents an active SSE connection
type streamableSSEStream struct {
	response *http.Response
	reader   *bufio.Reader
	messages chan []RpcMessage
	errors   chan error
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

	h := &StreamableHTTP{
		config: config,
		client: client,
	}

	// Initialize instance-specific message storage
	h.messageStore.messages = make([][]RpcMessage, 0)

	return h
}

// Connect establishes the HTTP transport connection.
// Not thread-safe - must not be called concurrently with other methods.
func (t *StreamableHTTP) Connect(ctx context.Context) error {
	if t.connectedState.Load() {
		return ErrAlreadyConnected
	}

	// For HTTP transport, we don't need to establish a persistent connection
	// We'll connect on-demand for each request
	t.connectedState.Store(true)

	return nil
}

// Close closes the HTTP transport connection.
// Not thread-safe - must not be called concurrently with other methods.
func (t *StreamableHTTP) Close() error {
	if !t.connectedState.Load() {
		return nil
	}
	t.connectedState.Store(false)

	// Close all SSE streams
	t.sseStreams.Range(func(key, value interface{}) bool {
		if stream, ok := value.(*streamableSSEStream); ok {
			close(stream.done)
			stream.response.Body.Close()
		}
		t.sseStreams.Delete(key)
		return true
	})

	// Send DELETE request if we have a session ID
	if t.sessionID != "" {
		req, err := http.NewRequest("DELETE", t.config.URL, nil)
		if err == nil {
			t.addHeaders(req)
			req.Header.Set("Mcp-Session-Id", t.sessionID)
			t.client.Do(req)
		}
	}

	return nil
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
		// Regular JSON response
		defer resp.Body.Close()

		// Parse response
		var response interface{}
		if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
			return NewTransportError("streamable-http", "decode response", err)
		}

		// Convert to message and store for Receive
		data, err := json.Marshal(response)
		if err != nil {
			return err
		}
		var responseMsg RpcMessage
		if err := json.Unmarshal(data, &responseMsg); err != nil {
			return err
		}

		// Store message for immediate retrieval
		t.storeMessages([]RpcMessage{responseMsg})

		return nil
	}
}

// handleSSEResponse handles a Server-Sent Events response
func (t *StreamableHTTP) handleSSEResponse(resp *http.Response) error {
	stream := &streamableSSEStream{
		response: resp,
		reader:   bufio.NewReader(resp.Body),
		messages: make(chan []RpcMessage, 100),
		errors:   make(chan error, 1),
		done:     make(chan struct{}),
	}

	// Store stream using sync.Map with atomic counter for unique ID
	streamID := fmt.Sprintf("stream-%d", t.streamCounter.Add(1))
	t.sseStreams.Store(streamID, stream)

	// Start reading SSE events
	go t.readSSEStream(streamID, stream)

	return nil
}

// readSSEStream reads events from an SSE stream
func (t *StreamableHTTP) readSSEStream(streamID string, stream *streamableSSEStream) {
	defer func() {
		stream.response.Body.Close()
		t.sseStreams.Delete(streamID)
		close(stream.messages)
		close(stream.errors)
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
				stream.errors <- err
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
					stream.errors <- fmt.Errorf("failed to parse SSE event: %w", err)
				} else {
					stream.messages <- []RpcMessage{msg}
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

// Receive receives JSON-RPC messages.
// Thread-safe due to internal synchronization.
func (t *StreamableHTTP) Receive() ([]RpcMessage, error) {
	if !t.connectedState.Load() {
		return nil, ErrNotConnected
	}

	// Check if we have any stored messages
	msgs := t.getStoredMessages()
	if len(msgs) > 0 {
		return msgs, nil
	}

	// Check SSE streams
	streams := make([]*streamableSSEStream, 0)
	t.sseStreams.Range(func(key, value interface{}) bool {
		if stream, ok := value.(*streamableSSEStream); ok {
			streams = append(streams, stream)
		}
		return true
	})

	if len(streams) == 0 {
		// No active streams, open a GET request for SSE
		if err := t.openSSEStream(); err != nil {
			return nil, err
		}

		// Get the new stream
		t.sseStreams.Range(func(key, value interface{}) bool {
			if stream, ok := value.(*streamableSSEStream); ok {
				streams = append(streams, stream)
				return false // Stop after first stream
			}
			return true
		})
	}

	// Wait for messages from any stream
	for _, stream := range streams {
		select {
		case msgs := <-stream.messages:
			return msgs, nil
		case err := <-stream.errors:
			return nil, err
		default:
			// try next stream
		}
	}

	// No messages available
	return nil, nil
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

// storeMessages stores messages for retrieval (instance-specific)
func (t *StreamableHTTP) storeMessages(msgs []RpcMessage) {
	t.messageStore.Lock()
	defer t.messageStore.Unlock()
	t.messageStore.messages = append(t.messageStore.messages, msgs)
}

// getStoredMessages retrieves stored messages (instance-specific)
func (t *StreamableHTTP) getStoredMessages() []RpcMessage {
	t.messageStore.Lock()
	defer t.messageStore.Unlock()

	if len(t.messageStore.messages) == 0 {
		return nil
	}

	msgs := t.messageStore.messages[0]
	t.messageStore.messages = t.messageStore.messages[1:]
	return msgs
}
