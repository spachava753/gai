package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

// HTTPConfig contains configuration for HTTP transport
type HTTPConfig struct {
	Config

	// URL is the MCP endpoint URL
	URL string `json:"url"`

	// Headers to include in requests
	Headers map[string]string `json:"headers,omitempty"`

	// HTTPClient allows using a custom HTTP client
	HTTPClient *http.Client `json:"-"`
}

// HTTP implements Transport using HTTP with SSE support.
// Thread Safety: This transport has mixed concurrency requirements:
// - Connect/Close must not be called concurrently
// - Send/Receive are synchronized internally due to HTTP requests and SSE streams
// - SSE stream management uses sync.Map for safe concurrent access
type HTTP struct {
	config    HTTPConfig
	client    *http.Client
	sessionID string

	// State that needs synchronization
	connectionState struct {
		sync.RWMutex
		connected bool
	}

	// SSE stream management using sync.Map for concurrent access
	sseStreams sync.Map // map[string]*sseStream

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

	// Backwards compatibility with old HTTP+SSE transport
	legacyState struct {
		sync.RWMutex
		legacySse    bool
		postEndpoint string // Separate POST endpoint for old protocol
	}
}

// sseStream represents an active SSE connection
type sseStream struct {
	response *http.Response
	reader   *bufio.Reader
	messages chan []RpcMessage
	errors   chan error
	done     chan struct{}
}

// NewHTTP creates a new HTTP transport
func NewHTTP(config HTTPConfig) *HTTP {
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{
			Timeout: 30 * time.Second,
		}
	}

	h := &HTTP{
		config: config,
		client: client,
	}

	// Initialize instance-specific message storage
	h.messageStore.messages = make([][]RpcMessage, 0)

	return h
}

// Connect establishes the HTTP transport connection.
// Not thread-safe - must not be called concurrently with other methods.
func (t *HTTP) Connect(ctx context.Context) error {
	t.connectionState.Lock()
	defer t.connectionState.Unlock()

	if t.connectionState.connected {
		return ErrAlreadyConnected
	}

	// For HTTP transport, we don't need to establish a persistent connection
	// We'll connect on-demand for each request
	t.connectionState.connected = true

	return nil
}

// Close closes the HTTP transport connection.
// Not thread-safe - must not be called concurrently with other methods.
func (t *HTTP) Close() error {
	t.connectionState.Lock()
	if !t.connectionState.connected {
		t.connectionState.Unlock()
		return nil
	}
	t.connectionState.connected = false
	t.connectionState.Unlock()

	// Close all SSE streams
	t.sseStreams.Range(func(key, value interface{}) bool {
		if stream, ok := value.(*sseStream); ok {
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
func (t *HTTP) Send(msg RpcMessage) error {
	t.connectionState.RLock()
	connected := t.connectionState.connected
	t.connectionState.RUnlock()

	if !connected {
		return ErrNotConnected
	}

	return t.sendPost([]RpcMessage{msg})
}

// SendBatch sends a batch of JSON-RPC messages
func (t *HTTP) SendBatch(messages []RpcMessage) error {
	t.connectionState.RLock()
	connected := t.connectionState.connected
	t.connectionState.RUnlock()

	if !connected {
		return ErrNotConnected
	}

	var hasResp bool
	for _, msg := range messages {
		// Check if it's a response (has ID and Result/Error but no Method)
		if msg.ID != nil && (msg.Result != nil || msg.Error != nil) && msg.Method == "" {
			hasResp = true
		} else if msg.Method != "" { // Request or notification
			if hasResp {
				return NewTransportError(
					"http",
					"send batch",
					errors.New("cannot mix responses in a batch request with requests and notifications"),
				)
			}
		}
	}

	return t.sendPost(messages)
}

// sendPost handles the actual POST sending with retry capability
func (t *HTTP) sendPost(messages []RpcMessage) error {
	var body []byte
	var err error

	if len(messages) == 1 {
		// Single message
		body, err = json.Marshal(messages[0])
	} else {
		// Batch
		body, err = json.Marshal(messages)
	}

	if err != nil {
		return fmt.Errorf("failed to marshal messages: %w", err)
	}

	// Determine which URL to use
	postURL := t.config.URL

	t.legacyState.RLock()
	legacy := t.legacyState.legacySse
	endpoint := t.legacyState.postEndpoint
	t.legacyState.RUnlock()

	if legacy && endpoint != "" {
		postURL = endpoint
	} else if legacy {
		panic("expected custom post url for legacy SSE")
	}

	req, err := http.NewRequest(http.MethodPost, postURL, bytes.NewReader(body))
	if err != nil {
		return NewTransportError("http", "create request", err)
	}

	t.addHeaders(req)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")

	resp, err := t.client.Do(req)
	if err != nil {
		return NewTransportError("http", "send request", err)
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
		return NewAuthenticationError("HTTP 401 Unauthorized")
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
		if len(messages) == 1 {
			msg := messages[0]

			t.legacyState.RLock()
			isLegacy := t.legacyState.legacySse
			t.legacyState.RUnlock()

			if msg.Method == "initialize" &&
				(resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMethodNotAllowed) &&
				!isLegacy {
				resp.Body.Close()

				// Try backwards compatibility mode
				if err = t.initializeLegacySse(); err != nil {
					return err
				}

				// Retry the request with old transport settings
				return t.sendPost(messages)
			}
		}

		// Error response
		defer resp.Body.Close()
		var errResp RpcMessage
		b, _ := io.ReadAll(resp.Body)
		if err = json.Unmarshal(b, &errResp); err != nil {
			return NewTransportError("http", "decode error response",
				fmt.Errorf("HTTP %d", resp.StatusCode))
		}
		if errResp.Error != nil {
			return NewProtocolError(errResp.Error.Code, errResp.Error.Message, errResp.Error.Data)
		}
		return NewTransportError("http", "request failed",
			fmt.Errorf("HTTP %d", resp.StatusCode))
	}

	// Check if this is the initialize response
	if len(messages) == 1 {
		msg := messages[0]
		if msg.Method == "initialize" && msg.ID != nil {
			// Extract session ID from response headers
			if sessionID := resp.Header.Get("Mcp-Session-Id"); sessionID != "" {
				t.sessionID = sessionID
			}
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
			return NewTransportError("http", "decode response", err)
		}

		// Convert to messages and store for Receive
		var msgs []RpcMessage

		// Check if it's a batch response
		if arr, ok := response.([]interface{}); ok {
			for _, item := range arr {
				data, err := json.Marshal(item)
				if err != nil {
					return err
				}
				var msg RpcMessage
				if err := json.Unmarshal(data, &msg); err != nil {
					return err
				}
				msgs = append(msgs, msg)
			}
		} else {
			data, err := json.Marshal(response)
			if err != nil {
				return err
			}
			var msg RpcMessage
			if err := json.Unmarshal(data, &msg); err != nil {
				return err
			}
			msgs = []RpcMessage{msg}
		}

		// Store messages for immediate retrieval
		t.storeMessages(msgs)

		return nil
	}
}

// handleSSEResponse handles a Server-Sent Events response
func (t *HTTP) handleSSEResponse(resp *http.Response) error {
	stream := &sseStream{
		response: resp,
		reader:   bufio.NewReader(resp.Body),
		messages: make(chan []RpcMessage, 100),
		errors:   make(chan error, 1),
		done:     make(chan struct{}),
	}

	// Store stream using sync.Map
	streamID := fmt.Sprintf("stream-%d", time.Now().UnixNano())
	t.sseStreams.Store(streamID, stream)

	// Start reading SSE events
	go t.readSSEStream(streamID, stream)

	return nil
}

// readSSEStream reads events from an SSE stream
func (t *HTTP) readSSEStream(streamID string, stream *sseStream) {
	defer func() {
		stream.response.Body.Close()
		t.sseStreams.Delete(streamID)
		close(stream.messages)
		close(stream.errors)
	}()

	var eventData bytes.Buffer
	var eventID string
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
				stream.errors <- err
			}
			return
		}

		line = strings.TrimSpace(line)

		if line == "" {
			// End of event
			if eventData.Len() > 0 {
				t.legacyState.RLock()
				isLegacy := t.legacyState.legacySse
				t.legacyState.RUnlock()

				// Check if this is an endpoint event (old protocol)
				if eventType == "endpoint" && isLegacy {
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

					// TODO: should we check if t.postEndpoint is already set? Theoretically, should not be an issue for a well-behaving server
					t.legacyState.Lock()
					t.legacyState.postEndpoint = endpointData
					t.legacyState.Unlock()
				} else {
					// Parse the event data as a JSON-RPC message
					var msg RpcMessage
					if err := json.Unmarshal(eventData.Bytes(), &msg); err != nil {
						stream.errors <- fmt.Errorf("failed to parse SSE event: %w", err)
					} else {
						stream.messages <- []RpcMessage{msg}
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
				eventType = ""
			}
			continue
		}

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			eventData.WriteString(data)
		} else if strings.HasPrefix(line, "id: ") {
			eventID = strings.TrimPrefix(line, "id: ")
		} else if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
		}
		// Ignore other fields like retry:, etc.
	}
}

// Receive receives JSON-RPC messages.
// Thread-safe due to internal synchronization.
func (t *HTTP) Receive() ([]RpcMessage, error) {
	t.connectionState.RLock()
	connected := t.connectionState.connected
	t.connectionState.RUnlock()

	if !connected {
		return nil, ErrNotConnected
	}

	// Check if we have any stored messages
	msgs := t.getStoredMessages()
	if len(msgs) > 0 {
		return msgs, nil
	}

	// Check SSE streams
	streams := make([]*sseStream, 0)
	t.sseStreams.Range(func(key, value interface{}) bool {
		if stream, ok := value.(*sseStream); ok {
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
			if stream, ok := value.(*sseStream); ok {
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
func (t *HTTP) openSSEStream() error {
	req, err := http.NewRequest(http.MethodGet, t.config.URL, nil)
	if err != nil {
		return NewTransportError("http", "create GET request", err)
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
		return NewTransportError("http", "send GET request", err)
	}

	if resp.StatusCode == 405 {
		// Method not allowed, server doesn't support SSE via GET
		resp.Body.Close()
		return nil
	}

	if resp.StatusCode >= 400 {
		resp.Body.Close()
		return NewTransportError("http", "GET request failed",
			fmt.Errorf("HTTP %d", resp.StatusCode))
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "text/event-stream") {
		resp.Body.Close()
		return NewTransportError("http", "unexpected content type",
			fmt.Errorf("expected text/event-stream, got %s", contentType))
	}

	return t.handleSSEResponse(resp)
}

// addHeaders adds common headers to a request
func (t *HTTP) addHeaders(req *http.Request) {
	// Add custom headers
	for k, v := range t.config.Headers {
		req.Header.Set(k, v)
	}

	// Add session ID if we have one
	if t.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", t.sessionID)
	}

	// Add protocol version only for new http transport
	t.legacyState.RLock()
	isLegacy := t.legacyState.legacySse
	t.legacyState.RUnlock()

	if !isLegacy {
		req.Header.Set("MCP-Protocol-Version", ProtocolVersion)
	}
}

// initializeLegacySse handles backwards compatibility with old HTTP+SSE transport
func (t *HTTP) initializeLegacySse() error {
	t.legacyState.Lock()
	t.legacyState.legacySse = true
	t.legacyState.Unlock()

	// Issue a GET request to establish SSE connection
	req, err := http.NewRequest(http.MethodGet, t.config.URL, nil)
	if err != nil {
		return NewTransportError("http", "create GET request for legacy http+sse transport", err)
	}

	t.addHeaders(req)
	req.Header.Set("Accept", "text/event-stream")

	resp, err := t.client.Do(req)
	if err != nil {
		return NewTransportError("http", "send GET request for legacy http+sse transport", err)
	}

	if resp.StatusCode >= 400 {
		resp.Body.Close()
		return NewTransportError("http", "GET request failed for legacy http+sse transport",
			fmt.Errorf("HTTP %d", resp.StatusCode))
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "text/event-stream") {
		resp.Body.Close()
		return NewTransportError("http", "unexpected content type for legacy http+sse transport",
			fmt.Errorf("expected text/event-stream, got %s", contentType))
	}

	// Handle SSE response
	if err = t.handleSSEResponse(resp); err != nil {
		return err
	}

	// Wait for endpoint event
	endpointChan := make(chan struct{})
	errChan := make(chan error)

	go func() {
		timeout := time.After(5 * time.Second)
		ticker := time.NewTicker(50 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-timeout:
				errChan <- fmt.Errorf("timeout waiting for endpoint event")
				return
			case <-ticker.C:
				t.legacyState.RLock()
				endpoint := t.legacyState.postEndpoint
				t.legacyState.RUnlock()

				if endpoint != "" {
					endpointChan <- struct{}{}
					return
				}
			}
		}
	}()

	select {
	case <-endpointChan:
		// Successfully got endpoint, continue
		return nil
	case err := <-errChan:
		return NewTransportError("http", "initialize legacy http+sse transport", err)
	}
}

// storeMessages stores messages for retrieval (instance-specific)
func (t *HTTP) storeMessages(msgs []RpcMessage) {
	t.messageStore.Lock()
	defer t.messageStore.Unlock()
	t.messageStore.messages = append(t.messageStore.messages, msgs)
}

// getStoredMessages retrieves stored messages (instance-specific)
func (t *HTTP) getStoredMessages() []RpcMessage {
	t.messageStore.Lock()
	defer t.messageStore.Unlock()

	if len(t.messageStore.messages) == 0 {
		return nil
	}

	msgs := t.messageStore.messages[0]
	t.messageStore.messages = t.messageStore.messages[1:]
	return msgs
}

// IsConnected returns whether the transport is connected
func (t *HTTP) Connected() bool {
	t.connectionState.RLock()
	defer t.connectionState.RUnlock()
	return t.connectionState.connected
}
