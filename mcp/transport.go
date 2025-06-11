// This file provides transport implementations for the MCP client.
package mcp

import (
	"context"
)

// Transport defines the interface for MCP transport implementations
type Transport interface {
	// Connect establishes the transport connection
	Connect(ctx context.Context) error

	// Close closes the transport connection
	Close() error

	// Send sends a JSON-RPC message
	Send(msg RpcMessage) error

	// SendBatch sends a batch of JSON-RPC messages
	SendBatch(messages []RpcMessage) error

	// Receive receives JSON-RPC messages (may be a batch)
	Receive() ([]RpcMessage, error)

	// Connected returns whether the transport is connected
	Connected() bool
}

// Config contains common configuration for transports
type Config struct {
	// Timeout is the default timeout for operations
	Timeout int `json:"timeout,omitempty"`
}
