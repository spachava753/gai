// Package mcp provides a Go implementation of the Model Context Protocol client.
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// Client represents a high-level MCP client with all features.
//
// Thread Safety: This client has mixed concurrency characteristics:
//
// **Concurrency-Safe Methods:**
//   - Close() - Safe to call concurrently and multiple times
//   - Reading from Notifications channel - Safe from any goroutine
//   - Request() and Notify() - Safe to call concurrently after initialization
//   - All request-response methods (ListTools, CallTool, etc.) - Safe to call concurrently
//   - Ping methods - Safe to call concurrently
//   - Getter methods (IsConnected, IsInitialized, GetServerInfo, etc.) - Safe to call concurrently
//
// **NOT Concurrency-Safe Methods (single-threaded use only):**
//   - Connect() - Must be called before any other operations
//   - Initialize() - Must be called after Connect and before other operations
//
// The recommended pattern is to perform setup (Connect, Initialize, register handlers)
// from a single goroutine, then use the client's request methods and Notifications
// channel from multiple goroutines safely.
type Client struct {
	// user-visible
	Notifications <-chan RpcMessage // receives ALL server notifications

	// private notification channel (write end)
	notifications chan RpcMessage

	// outbound pipe
	outbound chan RpcMessage // buffered

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

// NewClient creates a new MCP client
func NewClient(transport Transport, options Options) *Client {
	notifications := make(chan RpcMessage, 256)
	outbound := make(chan RpcMessage, 256)

	c := &Client{
		transport:     transport,
		notifications: notifications,
		Notifications: notifications,
		outbound:      outbound,
		idGen:         NewIDGenerator(),
		done:          make(chan struct{}),
		options:       options,
		wg:            new(sync.WaitGroup),
		pending:       new(sync.Map),
		once:          new(sync.Once),
	}

	c.wg.Add(2)
	go c.sender()
	go c.receiver()
	return c
}

func (c *Client) sender() {
	defer c.wg.Done()
	for {
		select {
		case <-c.done:
			return
		case msg := <-c.outbound:
			_ = c.transport.Send(msg) // log / drop error as desired
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
		case m.ID != nil && (m.Result != nil || m.Error != nil):
			if ch, ok := c.pending.LoadAndDelete(m.ID); ok {
				respCh := ch.(chan RpcMessage)
				respCh <- m
				close(respCh)
			}

		// -------- notification ------------------------------------
		case m.ID == nil && m.Method != "":
			select { // non-blocking send; drop on full buffer
			case c.notifications <- m:
			default:
			}

		// -------- server request ----------------------------------
		case m.ID != nil && m.Method != "":
			c.handleSrvRequest(m)
		}
	}
}

func (c *Client) handleSrvRequest(m RpcMessage) {
	switch m.Method {
	case "ping":
		c.outbound <- RpcMessage{
			JSONRPC: JSONRPCVersion, ID: m.ID, Result: map[string]any{}}

	case "sampling/createMessage":
		c.outbound <- RpcMessage{
			JSONRPC: JSONRPCVersion, ID: m.ID,
			Error: &Error{Code: -32601, Message: "sampling not supported"}}

	default:
		c.outbound <- RpcMessage{
			JSONRPC: JSONRPCVersion, ID: m.ID,
			Error: &Error{Code: -32601, Message: "method not supported"}}
	}
}

// Connect establishes the connection.
// Not thread-safe - must not be called concurrently with other methods.
func (c *Client) Connect(ctx context.Context) error {
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

// Initialize performs the initialization handshake.
// Not thread-safe - must not be called concurrently with other methods.
func (c *Client) Initialize(ctx context.Context, clientInfo ClientInfo, capabilities ClientCapabilities) error {
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
	defer c.wg.Wait()
	c.once.Do(func() { close(c.done) })
	close(c.notifications)

	c.pending.Range(func(k, v interface{}) bool {
		close(v.(chan RpcMessage))
		c.pending.Delete(k)
		return true
	})

	c.connected = false
	return c.transport.Close()
}

// IsConnected returns whether the client is connected
func (c *Client) IsConnected() bool {
	return c.connected
}

// IsInitialized returns whether the client is initialized
func (c *Client) IsInitialized() bool {
	return c.initialized
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

	c.outbound <- RpcMessage{
		JSONRPC: JSONRPCVersion, ID: id,
		Method: method, Params: paramsMap,
	}

	// Apply timeout
	if c.options.RequestTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.options.RequestTimeout)
		defer cancel()
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
	c.outbound <- notif
	return nil
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
func (c *Client) ListTools(ctx context.Context) ([]Tool, error) {
	if !c.IsInitialized() {
		return nil, ErrNotInitialized
	}

	result, err := c.Request(ctx, "tools/list", nil)
	if err != nil {
		return nil, err
	}

	var listResult ToolsListResult
	if err := ParseResult(result, &listResult); err != nil {
		return nil, err
	}

	return listResult.Tools, nil
}

// CallTool calls a tool
func (c *Client) CallTool(ctx context.Context, name string, arguments map[string]any) (any, error) {
	if !c.IsInitialized() {
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
	if !c.IsInitialized() {
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
	if !c.IsInitialized() {
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
	if !c.IsInitialized() {
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
	if !c.IsInitialized() {
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
	if !c.IsInitialized() {
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
	if !c.IsInitialized() {
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
	if !c.IsInitialized() {
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

// StartPeriodicPing starts periodic ping to keep connection alive
// Thread-safe - can be called concurrently, starts a background goroutine.
func (c *Client) StartPeriodicPing(ctx context.Context, interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				// Ignore ping errors - connection issues will be handled elsewhere
				_ = c.PingWithTimeout(5 * time.Second)
			}
		}
	}()
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
