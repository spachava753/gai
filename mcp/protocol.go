package mcp

import (
	"context"
	"fmt"
)

// Transport defines the interface for MCP transport implementations
type Transport interface {
	// Connect establishes the transport connection
	Connect(ctx context.Context) error

	// Close closes the transport connection
	Close() error

	// Send sends a JSON-RPC message
	Send(msg RpcMessage) error

	// Receive receives JSON-RPC messages
	Receive() ([]RpcMessage, error)
}

// ProtocolVersion is the version of the MCP protocol supported by this implementation
const ProtocolVersion = "2025-03-26"

// JSONRPCVersion is the JSON-RPC version used
const JSONRPCVersion = "2.0"

// RequestID can be either a string or number, but we are hardcoding it to be a string for now
type RequestID = string

// RpcMessage represents a JSON-RPC message, which can be a request, response or a notification
type RpcMessage struct {
	JSONRPC string         `json:"jsonrpc"`
	ID      RequestID      `json:"id,omitempty"`
	Method  string         `json:"method"`
	Params  map[string]any `json:"params,omitempty"`
	Result  map[string]any `json:"result,omitempty"`
	Error   *Error         `json:"error,omitempty"`
}

// Error represents a JSON-RPC error
type Error struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

func (e Error) Error() string {
	if e.Data != nil {
		return fmt.Sprintf("JSON-RPC error %d: %s (data: %v)", e.Code, e.Message, e.Data)
	}
	return fmt.Sprintf("JSON-RPC error %d: %s", e.Code, e.Message)
}

// Role represents the role in a message (for sampling)
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// ContentType represents the type of content
type ContentType string

const (
	ContentTypeText  ContentType = "text"
	ContentTypeImage ContentType = "image"
	ContentTypeAudio ContentType = "audio"
)

// Content represents content in a message
type Content struct {
	Type     ContentType `json:"type"`
	Text     string      `json:"text,omitempty"`
	Data     string      `json:"data,omitempty"`
	MimeType string      `json:"mimeType,omitempty"`
}

// Message represents a message in sampling
type Message struct {
	Role    Role    `json:"role"`
	Content Content `json:"content"`
}

// Capability represents a capability with optional sub-capabilities
type Capability struct {
	ListChanged bool `json:"listChanged,omitempty"`
	Subscribe   bool `json:"subscribe,omitempty"`
}

// ClientCapabilities represents the capabilities of a client
type ClientCapabilities struct {
	// TODO: support roots client capability
	//Roots        *Capability            `json:"roots,omitempty"`
	// TODO: support roots client capability
	//Sampling     *Capability            `json:"sampling,omitempty"`
	Experimental map[string]interface{} `json:"experimental,omitempty"`
}

// ServerCapabilities represents the capabilities of a server
type ServerCapabilities struct {
	Prompts      *Capability            `json:"prompts,omitempty"`
	Resources    *Capability            `json:"resources,omitempty"`
	Tools        *Capability            `json:"tools,omitempty"`
	Logging      *Capability            `json:"logging,omitempty"`
	Experimental map[string]interface{} `json:"experimental,omitempty"`
}

// ClientInfo represents client implementation information
type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// ServerInfo represents server implementation information
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// InitializeParams represents parameters for the initialize request
type InitializeParams struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    ClientCapabilities `json:"capabilities"`
	ClientInfo      ClientInfo         `json:"clientInfo"`
}

// InitializeResult represents the result of an initialize request
type InitializeResult struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    ServerCapabilities `json:"capabilities"`
	ServerInfo      ServerInfo         `json:"serverInfo"`
	Instructions    string             `json:"instructions,omitempty"`
}

// Root represents a filesystem root
type Root struct {
	URI  string `json:"uri"`
	Name string `json:"name,omitempty"`
}

// RootsListResult represents the result of a roots/list request
type RootsListResult struct {
	Roots []Root `json:"roots"`
}

// tool represents a tool definition
type tool struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	InputSchema InputSchema `json:"inputSchema"`
}

// InputSchema represents the schema for tool input
type InputSchema struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	Required   []string               `json:"required,omitempty"`
}

// toolsListResult represents the result of a tools/list request
type toolsListResult struct {
	Tools []tool `json:"tools"`
}

// Resource represents a resource
type Resource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
}

// ResourcesListResult represents the result of a resources/list request
type ResourcesListResult struct {
	Resources []Resource `json:"resources"`
}

// ResourceContent represents the content of a resource
type ResourceContent struct {
	URI      string `json:"uri"`
	MimeType string `json:"mimeType,omitempty"`
	Text     string `json:"text,omitempty"`
	Blob     string `json:"blob,omitempty"`
}

// ResourcesReadParams represents parameters for resources/read request
type ResourcesReadParams struct {
	URI string `json:"uri"`
}

// ResourcesReadResult represents the result of a resources/read request
type ResourcesReadResult struct {
	Contents []ResourceContent `json:"contents"`
}

// Prompt represents a prompt template
type Prompt struct {
	Name        string           `json:"name"`
	Description string           `json:"description,omitempty"`
	Arguments   []PromptArgument `json:"arguments,omitempty"`
}

// PromptArgument represents an argument for a prompt
type PromptArgument struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required,omitempty"`
}

// PromptsListResult represents the result of a prompts/list request
type PromptsListResult struct {
	Prompts []Prompt `json:"prompts"`
}

// PromptsGetParams represents parameters for prompts/get request
type PromptsGetParams struct {
	Name      string            `json:"name"`
	Arguments map[string]string `json:"arguments,omitempty"`
}

// PromptsGetResult represents the result of a prompts/get request
type PromptsGetResult struct {
	Description string    `json:"description,omitempty"`
	Messages    []Message `json:"messages"`
}

// ModelPreferences represents model selection preferences
type ModelPreferences struct {
	Hints                []ModelHint `json:"hints,omitempty"`
	CostPriority         float64     `json:"costPriority,omitempty"`
	SpeedPriority        float64     `json:"speedPriority,omitempty"`
	IntelligencePriority float64     `json:"intelligencePriority,omitempty"`
}

// ModelHint represents a hint for model selection
type ModelHint struct {
	Name string `json:"name"`
}

// CreateMessageParams represents parameters for sampling/createMessage
type CreateMessageParams struct {
	Messages         []Message         `json:"messages"`
	ModelPreferences *ModelPreferences `json:"modelPreferences,omitempty"`
	SystemPrompt     string            `json:"systemPrompt,omitempty"`
	MaxTokens        int               `json:"maxTokens,omitempty"`
}

// CreateMessageResult represents the result of sampling/createMessage
type CreateMessageResult struct {
	Role       Role    `json:"role"`
	Content    Content `json:"content"`
	Model      string  `json:"model"`
	StopReason string  `json:"stopReason,omitempty"`
}

// ProgressParams represents parameters for progress notifications
type ProgressParams struct {
	ProgressToken string  `json:"progressToken"`
	Progress      float64 `json:"progress"`
	Total         float64 `json:"total,omitempty"`
	Message       string  `json:"message,omitempty"`
}

// CancelledParams represents parameters for cancellation notifications
type CancelledParams struct {
	RequestID RequestID `json:"requestId"`
	Reason    string    `json:"reason,omitempty"`
}

// Meta represents metadata that can be included in requests
type Meta struct {
	ProgressToken string `json:"progressToken,omitempty"`
}
