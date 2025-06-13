// Package mcp provides a Go implementation of the Model Context Protocol client.
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/spachava753/gai"
	"sync"
	"time"
)

// Options contains options for creating a client
type Options struct {
	// Handler for incoming messages
	MessageHandler func(msg interface{})

	// Handler for errors
	ErrorHandler func(err error)

	// Enable logging
	EnableLogging bool

	// Log handler
	LogHandler func(level, message string)
}

// DefaultOptions returns default client options
func DefaultOptions() Options {
	return Options{
		MessageHandler: func(msg interface{}) {
			// Default: ignore unhandled messages
		},
		ErrorHandler: func(err error) {
			// Default: ignore errors
		},
	}
}

// Client represents a high-level MCP client with all features.
//
// Thread Safety: This client is designed for concurrent use after initialization:
//
// **Concurrency-Safe Methods (safe to call from multiple goroutines):**
//   - Close() - Safe to call concurrently and multiple times
//   - Reading from Notifications() channel - Safe from any goroutine
//   - Request() and Notify() - Safe to call concurrently
//   - All request-response methods (ListTools, CallTool, etc.) - Safe to call concurrently
//   - Ping methods - Safe to call concurrently
//   - Getter methods (IsConnected, GetServerInfo, etc.) - Safe to call concurrently
//
// **Initialization:**
//
//	The client automatically connects and initializes during construction via NewClient().
//	No additional setup steps are required - the returned client is ready to use immediately.
//
// **Recommended usage pattern:**
//
//	```go
//	client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, options)
//	if err != nil {
//	    // Handle initialization error
//	    return err
//	}
//	defer client.Close()
//
//	// Client is now ready for concurrent use
//	tools, err := client.ListTools(ctx)
//	// ... use client methods safely from multiple goroutines
//	```
type Client struct {
	// private notification channel (write end)
	notifications chan RpcMessage

	// outbound pipe
	outbound chan struct {
		msg     RpcMessage
		errChan chan error
	}

	// request tracking
	pending *sync.Map // id -> chan RpcMessage

	// misc
	transport Transport
	idGen     *IDGenerator

	// goroutine coordination
	done chan struct{}   // close(done) => stop all internals
	once *sync.Once      // guards close(done)
	wg   *sync.WaitGroup // waits for sender + receiver

	// Options
	options Options

	// State - no mutex needed for most fields, single-threaded access assumed
	connected bool

	// Protocol version negotiated
	protocolVersion string

	// Server capabilities
	serverCapabilities ServerCapabilities

	// Server info
	serverInfo ServerInfo

	// Server instructions
	instructions string

	// Initialization state
	initialized bool
}

// NewClient creates a new MCP client and automatically connects and initializes it.
// This constructor handles all the setup steps (connect, initialize) internally, so the
// returned client is ready to use immediately.
//
// The client will automatically:
// 1. Connect to the transport
// 2. Perform the MCP initialization handshake
// 3. Start background goroutines for message handling
//
// Parameters:
//   - ctx: context for the connection and initialization operations
//   - transport: the transport to use for communication (stdio, HTTP, etc.)
//   - clientInfo: information about this client
//   - capabilities: client capabilities to negotiate with the server
//   - options: additional client options
//
// Returns a fully initialized and ready-to-use client, or an error if setup fails.
func NewClient(ctx context.Context, transport Transport, clientInfo ClientInfo, capabilities ClientCapabilities, options Options) (*Client, error) {
	notifications := make(chan RpcMessage, 256)
	outbound := make(chan struct {
		msg     RpcMessage
		errChan chan error
	}, 256)

	c := &Client{
		transport:     transport,
		notifications: notifications,
		outbound:      outbound,
		idGen:         NewIDGenerator(),
		done:          make(chan struct{}),
		options:       options,
		wg:            new(sync.WaitGroup),
		pending:       new(sync.Map),
		once:          new(sync.Once),
	}

	// Start background goroutines
	c.wg.Add(2)
	go c.sender()
	go c.receiver()

	// Connect to the transport
	if err := c.connect(ctx); err != nil {
		c.Close() // Clean up goroutines
		return nil, fmt.Errorf("failed to connect: %w", err)
	}

	// Initialize the MCP protocol
	if err := c.initialize(ctx, clientInfo, capabilities); err != nil {
		c.Close() // Clean up connection and goroutines
		return nil, fmt.Errorf("failed to initialize: %w", err)
	}

	return c, nil
}

func (c *Client) sender() {
	defer c.wg.Done()
	for {
		select {
		case <-c.done:
			return
		case msg := <-c.outbound:
			err := c.transport.Send(msg.msg)
			// try to send an error through the channel, if it is not nil
			if err != nil && msg.errChan != nil {
				select {
				case msg.errChan <- err:
				}
			}
			// close the channel if not nil, regardless of whether we used or not
			if msg.errChan != nil {
				close(msg.errChan)
			}
		}
	}
}

func (c *Client) receiver() {
	defer c.wg.Done()
	for {
		select {
		case <-c.done:
			return
		default:
		}

		msgs, err := c.transport.Receive()
		if err != nil {
			if c.options.ErrorHandler != nil {
				c.options.ErrorHandler(fmt.Errorf("receive error: %w", err))
			}
			continue
		}

		c.dispatch(msgs)
	}
}

func (c *Client) dispatch(msgs []RpcMessage) {
	for _, m := range msgs {
		switch {
		// -------- response ----------------------------------------
		case m.ID != "" && (m.Result != nil || m.Error != nil):
			if ch, ok := c.pending.LoadAndDelete(m.ID); ok {
				respCh := ch.(chan RpcMessage)
				respCh <- m
				close(respCh)
			}

		// -------- notification ------------------------------------
		case m.ID == "" && m.Method != "":
			select { // non-blocking send; drop on full buffer
			case c.notifications <- m:
			default:
			}

		// -------- server request ----------------------------------
		case m.ID != "" && m.Method != "":
			c.handleSrvRequest(m)
		}
	}
}

func (c *Client) handleSrvRequest(m RpcMessage) {
	switch m.Method {
	case "ping":
		c.outbound <- struct {
			msg     RpcMessage
			errChan chan error
		}{
			msg: RpcMessage{
				JSONRPC: JSONRPCVersion, ID: m.ID, Result: map[string]any{}},
		}

	case "sampling/createMessage":
		c.outbound <- struct {
			msg     RpcMessage
			errChan chan error
		}{
			msg: RpcMessage{
				JSONRPC: JSONRPCVersion, ID: m.ID,
				Error: &Error{Code: -32601, Message: "sampling not supported"}},
		}
	default:
		c.outbound <- struct {
			msg     RpcMessage
			errChan chan error
		}{
			msg: RpcMessage{
				JSONRPC: JSONRPCVersion, ID: m.ID,
				Error: &Error{Code: -32601, Message: "method not supported"}},
		}
	}
}

// connect establishes the connection.
func (c *Client) connect(ctx context.Context) error {
	if c.connected {
		return ErrAlreadyConnected
	}

	// Connect transport
	if err := c.transport.Connect(ctx); err != nil {
		return fmt.Errorf("transport connect failed: %w", err)
	}

	c.connected = true
	return nil
}

// initialize performs the initialization handshake.
func (c *Client) initialize(ctx context.Context, clientInfo ClientInfo, capabilities ClientCapabilities) error {
	if c.initialized {
		return ErrAlreadyInitialized
	}

	if !c.connected {
		return ErrNotConnected
	}

	// Prepare initialize params
	params := InitializeParams{
		ProtocolVersion: ProtocolVersion,
		Capabilities:    capabilities,
		ClientInfo:      clientInfo,
	}

	result, err := c.Request(ctx, "initialize", params)
	if err != nil {
		return fmt.Errorf("initialize request failed: %w", err)
	}

	// Parse result
	var initResult InitializeResult
	if err := ParseResult(result, &initResult); err != nil {
		return fmt.Errorf("failed to parse initialize result: %w", err)
	}

	// Check protocol version compatibility.
	// The client supports any version from the server that is less than or equal
	// to its own version, as the protocol is designed to be backward-compatible.
	if initResult.ProtocolVersion > ProtocolVersion {
		return NewVersionMismatchError(ProtocolVersion, initResult.ProtocolVersion)
	}

	// Store server info
	c.protocolVersion = initResult.ProtocolVersion
	c.serverCapabilities = initResult.Capabilities
	c.serverInfo = initResult.ServerInfo
	c.instructions = initResult.Instructions

	// Send initialized notification
	err = c.Notify("notifications/initialized", nil)
	if err != nil {
		return fmt.Errorf("failed to send initialized notification: %w", err)
	}

	c.initialized = true
	return nil
}

// Close closes the client connection.
// This method is safe to call concurrently and multiple times.
func (c *Client) Close() error {
	var err error
	c.once.Do(func() {
		defer c.wg.Wait()
		close(c.done)
		close(c.notifications)
		c.pending.Range(func(k, v interface{}) bool {
			close(v.(chan RpcMessage))
			c.pending.Delete(k)
			return true
		})

		c.connected = false
		err = c.transport.Close()
	})
	return err
}

// IsConnected returns whether the client is connected
func (c *Client) IsConnected() bool {
	return c.connected
}

// Request sends a raw request (for advanced use)
func (c *Client) Request(ctx context.Context, method string, params interface{}) (map[string]interface{}, error) {
	if !c.IsConnected() {
		return nil, ErrNotConnected
	}

	id := c.idGen.Generate()
	respCh := make(chan RpcMessage, 1)
	c.pending.Store(id, respCh)

	// Convert params to map
	paramsMap, err := toMap(params)
	if err != nil {
		c.pending.Delete(id)
		return nil, fmt.Errorf("failed to convert params: %w", err)
	}

	errChan := make(chan error, 1)
	c.outbound <- struct {
		msg     RpcMessage
		errChan chan error
	}{
		msg: RpcMessage{
			JSONRPC: JSONRPCVersion, ID: id,
			Method: method, Params: paramsMap,
		},
		errChan: errChan,
	}

	// we expect sender to receive our request, and always close the err channel,
	// regardless of whether an error was sent or not
	select {
	case err = <-errChan:
		if err != nil {
			c.pending.Delete(id)
			return nil, fmt.Errorf("failed to do request: %w", err)
		}
	}

	select {
	case <-ctx.Done():
		c.pending.Delete(id)
		// optional cancel notification
		_ = c.Notify("notifications/cancelled",
			CancelledParams{RequestID: id, Reason: ctx.Err().Error()})
		return nil, ctx.Err()
	case msg := <-respCh:
		if msg.Error != nil {
			return nil, NewProtocolError(msg.Error.Code,
				msg.Error.Message, msg.Error.Data)
		}
		return msg.Result, nil
	}
}

// Notify sends a notification (for advanced use)
func (c *Client) Notify(method string, params interface{}) error {
	if !c.IsConnected() {
		return ErrNotConnected
	}

	// Convert params to map
	paramsMap, err := toMap(params)
	if err != nil {
		return fmt.Errorf("failed to convert params: %w", err)
	}

	// Create notification
	notif := RpcMessage{
		JSONRPC: JSONRPCVersion,
		Method:  method,
		Params:  paramsMap,
	}

	// Send notification
	errChan := make(chan error)
	c.outbound <- struct {
		msg     RpcMessage
		errChan chan error
	}{
		msg:     notif,
		errChan: errChan,
	}
	return <-errChan
}

// GetServerInfo returns server information
func (c *Client) GetServerInfo() ServerInfo {
	return c.serverInfo
}

// GetServerCapabilities returns server capabilities
func (c *Client) GetServerCapabilities() ServerCapabilities {
	return c.serverCapabilities
}

// GetInstructions returns server instructions
func (c *Client) GetInstructions() string {
	return c.instructions
}

// Tool-related methods

// ListTools lists available tools
func (c *Client) ListTools(ctx context.Context) ([]gai.Tool, error) {
	if !c.initialized {
		return nil, ErrNotInitialized
	}

	result, err := c.Request(ctx, "tools/list", nil)
	if err != nil {
		return nil, err
	}

	var listResult toolsListResult
	if err := ParseResult(result, &listResult); err != nil {
		return nil, err
	}

	// Convert internal tool definitions to gai.tool for public API
	converted := make([]gai.Tool, len(listResult.Tools))
	for i, t := range listResult.Tools {
		converted[i], err = convertMCPToolToGAITool(t)
		if err != nil {
			return nil, fmt.Errorf("cannot convert tool %s to gai.Tool type", t.Name)
		}
	}

	return converted, nil
}

// CallTool calls a tool by name. The arguments map must conform to the tool's input schema.
// The return value is provider-defined and may be of any JSON-compatible type.
func (c *Client) CallTool(ctx context.Context, name string, arguments map[string]any) (any, error) {
	if !c.initialized {
		return nil, ErrNotInitialized
	}

	params := map[string]interface{}{
		"name":      name,
		"arguments": arguments,
	}

	result, err := c.Request(ctx, "tools/call", params)
	if err != nil {
		return nil, err
	}

	// Tools can return any type of result
	if content, ok := result["content"]; ok {
		return content, nil
	}

	return result, nil
}

// Resource-related methods

// ListResources lists available resources
func (c *Client) ListResources(ctx context.Context) ([]Resource, error) {
	if !c.initialized {
		return nil, ErrNotInitialized
	}

	result, err := c.Request(ctx, "resources/list", nil)
	if err != nil {
		return nil, err
	}

	var listResult ResourcesListResult
	if err := ParseResult(result, &listResult); err != nil {
		return nil, err
	}

	return listResult.Resources, nil
}

// ReadResource reads a resource
func (c *Client) ReadResource(ctx context.Context, uri string) ([]ResourceContent, error) {
	if !c.initialized {
		return nil, ErrNotInitialized
	}

	params := ResourcesReadParams{
		URI: uri,
	}

	result, err := c.Request(ctx, "resources/read", params)
	if err != nil {
		return nil, err
	}

	var readResult ResourcesReadResult
	if err := ParseResult(result, &readResult); err != nil {
		return nil, err
	}

	return readResult.Contents, nil
}

// SubscribeToResource subscribes to resource changes
func (c *Client) SubscribeToResource(ctx context.Context, uri string) error {
	if !c.initialized {
		return ErrNotInitialized
	}

	// Check if server supports subscriptions
	serverCaps := c.GetServerCapabilities()
	if serverCaps.Resources == nil || !serverCaps.Resources.Subscribe {
		return NewUnsupportedFeatureError("resource subscriptions", "server does not support this feature")
	}

	params := map[string]interface{}{
		"uri": uri,
	}

	result, err := c.Request(ctx, "resources/subscribe", params)
	if err != nil {
		return err
	}

	// Check if subscription was successful
	if success, ok := result["success"].(bool); !ok || !success {
		return fmt.Errorf("subscription failed")
	}

	return nil
}

// UnsubscribeFromResource unsubscribes from resource changes
func (c *Client) UnsubscribeFromResource(ctx context.Context, uri string) error {
	if !c.initialized {
		return ErrNotInitialized
	}

	params := map[string]interface{}{
		"uri": uri,
	}

	_, err := c.Request(ctx, "resources/unsubscribe", params)
	return err
}

// Prompt-related methods

// ListPrompts lists available prompts
// Thread-safe - can be called concurrently after initialization.
func (c *Client) ListPrompts(ctx context.Context) ([]Prompt, error) {
	if !c.initialized {
		return nil, ErrNotInitialized
	}

	result, err := c.Request(ctx, "prompts/list", nil)
	if err != nil {
		return nil, err
	}

	var listResult PromptsListResult
	if err := ParseResult(result, &listResult); err != nil {
		return nil, err
	}

	return listResult.Prompts, nil
}

// GetPrompt gets a prompt with arguments
// Thread-safe - can be called concurrently after initialization.
func (c *Client) GetPrompt(ctx context.Context, name string, arguments map[string]string) (*PromptsGetResult, error) {
	if !c.initialized {
		return nil, ErrNotInitialized
	}

	params := PromptsGetParams{
		Name:      name,
		Arguments: arguments,
	}

	result, err := c.Request(ctx, "prompts/get", params)
	if err != nil {
		return nil, err
	}

	var getResult PromptsGetResult
	if err := ParseResult(result, &getResult); err != nil {
		return nil, err
	}

	return &getResult, nil
}

// Sampling-related methods

// CreateMessage creates a message using LLM sampling
// Thread-safe - can be called concurrently after initialization.
func (c *Client) CreateMessage(ctx context.Context, params CreateMessageParams) (*CreateMessageResult, error) {
	if !c.initialized {
		return nil, ErrNotInitialized
	}

	result, err := c.Request(ctx, "sampling/createMessage", params)
	if err != nil {
		return nil, err
	}

	var createResult CreateMessageResult
	if err := ParseResult(result, &createResult); err != nil {
		return nil, err
	}

	return &createResult, nil
}

// Ping sends a ping request
// Thread-safe - can be called concurrently after initialization.
func (c *Client) Ping(ctx context.Context) error {
	_, err := c.Request(ctx, "ping", nil)
	return err
}

// PingWithTimeout sends a ping with a specific timeout
// Thread-safe - can be called concurrently after initialization.
func (c *Client) PingWithTimeout(timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return c.Ping(ctx)
}

// SetLoggingLevel sets the server logging level
// Thread-safe - can be called concurrently after initialization.
func (c *Client) SetLoggingLevel(ctx context.Context, level string) error {
	params := map[string]interface{}{
		"level": level,
	}

	_, err := c.Request(ctx, "logging/setLevel", params)
	return err
}

// CancelRequest cancels a specific request
// Thread-safe - can be called concurrently after initialization.
func (c *Client) CancelRequest(requestID RequestID, reason string) error {
	params := CancelledParams{
		RequestID: requestID,
		Reason:    reason,
	}

	return c.Notify("notifications/cancelled", params)
}

// Notifications is a read only channel that returns all notifications sent from the mcp server
// Thread-safe - can be called concurrently after initialization.
func (c *Client) Notifications() <-chan RpcMessage {
	return c.notifications
}

// GetIDGenerator returns the ID generator (for advanced use)
// Thread-safe - can be called concurrently.
func (c *Client) GetIDGenerator() interface{ Generate() RequestID } {
	return c.idGen
}

// toMap converts a value to a map
func toMap(v interface{}) (map[string]interface{}, error) {
	if v == nil {
		return nil, nil
	}

	// If already a map, return it
	if m, ok := v.(map[string]interface{}); ok {
		return m, nil
	}

	// Convert via JSON
	data, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}

	return result, nil
}

// ParseResult parses a result map into a struct
func ParseResult(result map[string]interface{}, v interface{}) error {
	data, err := json.Marshal(result)
	if err != nil {
		return err
	}

	return json.Unmarshal(data, v)
}
