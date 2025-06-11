package mcp

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/spachava753/gai"
)

// ToolAdapter adapts MCP tools to gai.Tool format
type ToolAdapter struct {
	client *Client
	tool   Tool
}

// NewToolAdapter creates a new adapter for an MCP tool
func NewToolAdapter(client *Client, tool Tool) *ToolAdapter {
	return &ToolAdapter{
		client: client,
		tool:   tool,
	}
}

// ToGaiTool converts the MCP tool to a gai.Tool
func (a *ToolAdapter) ToGaiTool() gai.Tool {
	// Convert MCP InputSchema to gai.InputSchema
	var gaiSchema gai.InputSchema

	if a.tool.InputSchema.Type == "object" {
		gaiSchema.Type = gai.Object
		gaiSchema.Properties = make(map[string]gai.Property)

		// Convert properties
		for name, prop := range a.tool.InputSchema.Properties {
			gaiSchema.Properties[name] = a.convertProperty(prop)
		}

		gaiSchema.Required = a.tool.InputSchema.Required
	}

	return gai.Tool{
		Name:        a.tool.Name,
		Description: a.tool.Description,
		InputSchema: gaiSchema,
	}
}

// convertProperty converts an MCP property to gai.Property
func (a *ToolAdapter) convertProperty(prop interface{}) gai.Property {
	propMap, ok := prop.(map[string]interface{})
	if !ok {
		return gai.Property{Type: gai.String}
	}

	var result gai.Property

	// Get type
	if typeStr, ok := propMap["type"].(string); ok {
		switch typeStr {
		case "string":
			result.Type = gai.String
		case "number":
			result.Type = gai.Number
		case "integer":
			result.Type = gai.Integer
		case "boolean":
			result.Type = gai.Boolean
		case "array":
			result.Type = gai.Array
		case "object":
			result.Type = gai.Object
		default:
			result.Type = gai.String
		}
	}

	// Get description
	if desc, ok := propMap["description"].(string); ok {
		result.Description = desc
	}

	// Get enum values
	if enum, ok := propMap["enum"].([]interface{}); ok {
		result.Enum = make([]string, len(enum))
		for i, v := range enum {
			result.Enum[i] = fmt.Sprintf("%v", v)
		}
	}

	// Handle nested properties for objects
	if result.Type == gai.Object {
		if props, ok := propMap["properties"].(map[string]interface{}); ok {
			result.Properties = make(map[string]gai.Property)
			for name, prop := range props {
				result.Properties[name] = a.convertProperty(prop)
			}
		}

		if required, ok := propMap["required"].([]interface{}); ok {
			result.Required = make([]string, len(required))
			for i, r := range required {
				result.Required[i] = r.(string)
			}
		}
	}

	// Handle array items
	if result.Type == gai.Array {
		if items, ok := propMap["items"].(map[string]interface{}); ok {
			itemProp := a.convertProperty(items)
			result.Items = &itemProp
		}
	}

	return result
}

// CreateCallback creates a gai.ToolCallback for this tool
func (a *ToolAdapter) CreateCallback() gai.ToolCallback {
	return &toolCallback{
		client:   a.client,
		toolName: a.tool.Name,
	}
}

// toolCallback implements gai.ToolCallback
type toolCallback struct {
	client   *Client
	toolName string
}

// Call executes the MCP tool
func (c *toolCallback) Call(ctx context.Context, parametersJSON json.RawMessage, toolCallID string) (gai.Message, error) {
	// Parse parameters
	var params map[string]interface{}
	if err := json.Unmarshal(parametersJSON, &params); err != nil {
		return gai.Message{}, fmt.Errorf("failed to parse parameters: %w", err)
	}

	// Call MCP tool
	result, err := c.client.CallTool(ctx, c.toolName, params)
	if err != nil {
		// Return error as tool result
		return gai.Message{
			Role: gai.ToolResult,
			Blocks: []gai.Block{
				{
					ID:           toolCallID,
					BlockType:    gai.Content,
					ModalityType: gai.Text,
					MimeType:     "text/plain",
					Content:      gai.Str(fmt.Sprintf("Error: %v", err)),
				},
			},
			ToolResultError: true,
		}, nil
	}

	// Convert result to string
	var resultStr string
	switch v := result.(type) {
	case string:
		resultStr = v
	case map[string]interface{}:
		// Check if it's a content block
		if content, ok := v["content"].(string); ok {
			resultStr = content
		} else {
			// JSON encode the result
			data, err := json.Marshal(v)
			if err != nil {
				return gai.Message{}, fmt.Errorf("failed to marshal result: %w", err)
			}
			resultStr = string(data)
		}
	default:
		resultStr = fmt.Sprintf("%v", result)
	}

	// Return successful result
	return gai.Message{
		Role: gai.ToolResult,
		Blocks: []gai.Block{
			{
				ID:           toolCallID,
				BlockType:    gai.Content,
				ModalityType: gai.Text,
				MimeType:     "text/plain",
				Content:      gai.Str(resultStr),
			},
		},
	}, nil
}

// RegisterMCPToolsWithGenerator registers all MCP tools with a gai generator
func RegisterMCPToolsWithGenerator(ctx context.Context, client *Client, toolGen *gai.ToolGenerator) error {
	// List available tools
	tools, err := client.ListTools(ctx)
	if err != nil {
		return fmt.Errorf("failed to list MCP tools: %w", err)
	}

	// Register each tool
	for _, tool := range tools {
		adapter := NewToolAdapter(client, tool)
		gaiTool := adapter.ToGaiTool()
		callback := adapter.CreateCallback()

		if err := toolGen.Register(gaiTool, callback); err != nil {
			return fmt.Errorf("failed to register tool %s: %w", tool.Name, err)
		}
	}

	return nil
}

// Example of using MCP tools with gai
func ExampleMCPWithGai() {
	// Create MCP client
	config := StdioConfig{
		Command: "npx",
		Args:    []string{"-y", "@modelcontextprotocol/server-time"},
	}
	transport := NewStdio(config)
	mcpClient := NewClient(transport, DefaultOptions())

	// Connect and initialize MCP
	ctx := context.Background()
	if err := mcpClient.Connect(ctx); err != nil {
		panic(err)
	}
	defer mcpClient.Close()

	if err := mcpClient.Initialize(ctx, ClientInfo{Name: "example", Version: "1.0"}, ClientCapabilities{}); err != nil {
		panic(err)
	}

	// Create gai generator (example with OpenAI)
	// openaiClient := openai.NewClient()
	// baseGen := gai.NewOpenAiGenerator(
	// 	openaiClient.Chat.Completions,
	// 	openai.ChatModelGPT4,
	// 	"You are a helpful assistant with access to various tools.",
	// )

	// Create tool generator
	// toolGen := &gai.ToolGenerator{
	// 	G: &baseGen,
	// }

	// Register MCP tools
	// if err := RegisterMCPToolsWithGenerator(ctx, mcpClient, toolGen); err != nil {
	// 	panic(err)
	// }

	// Now you can use the generator with MCP tools
	// dialog := gai.Dialog{
	// 	{
	// 		Role: gai.User,
	// 		Blocks: []gai.Block{
	// 			gai.TextBlock("What time is it?"),
	// 		},
	// 	},
	// }

	// completeDialog, err := toolGen.Generate(ctx, dialog, nil)
	// if err != nil {
	// 	panic(err)
	// }

	// The model will automatically use MCP tools to answer the question
}
