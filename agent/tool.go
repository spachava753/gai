package agent

import (
	"context"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/spachava753/gai"
)

// Tool pairs a provider-visible function declaration with the handler that
// executes requested calls.
type Tool struct {
	// Definition is sent to the generator on every model call.
	Definition gai.Tool
	// Handler executes validated calls to Definition.
	Handler ToolHandler
}

// ToolHandler executes one validated model-requested tool call. ToolRequest
// references are borrowed and must not be changed or retained after Execute.
type ToolHandler interface {
	// Execute returns a model-visible result or an error that stops the run. A
	// successful result belongs to the loop, which assigns call IDs to its blocks.
	Execute(context.Context, ToolRequest) (gai.Message, error)
}

// ToolHandlerFunc adapts a function to ToolHandler.
type ToolHandlerFunc func(context.Context, ToolRequest) (gai.Message, error)

// Execute calls f.
func (f ToolHandlerFunc) Execute(ctx context.Context, request ToolRequest) (gai.Message, error) {
	return f(ctx, request)
}

// ToolRequest contains borrowed references to the original block and decoded
// call passed to a handler.
type ToolRequest struct {
	// Block is the original provider-produced tool-call block.
	Block gai.Block
	// Call contains the validated name and arguments used for execution.
	Call gai.ToolCallInput
}

type executableTool struct {
	definition gai.Tool
	handler    ToolHandler
	schema     *jsonschema.Resolved
}
