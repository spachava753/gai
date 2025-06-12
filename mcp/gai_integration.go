package mcp

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/spachava753/gai"
)

// convertMCPToolToGAITool converts an MCP tool definition to a GAI tool definition
// This allows MCP tools to be registered with the tool generator
func convertMCPToolToGAITool(mcpTool tool) (gai.Tool, error) {
	// Create a basic GAI tool with name and description
	gaiTool := gai.Tool{
		Name:        mcpTool.Name,
		Description: mcpTool.Description,
		InputSchema: gai.InputSchema{
			Type: gai.Object, // Default type
		},
	}

	// Validate input schema type
	if mcpTool.InputSchema.Type != "object" {
		return gai.Tool{}, fmt.Errorf("unsupported schema type: %s (only 'object' is supported)", mcpTool.InputSchema.Type)
	}

	// Copy required fields directly
	gaiTool.InputSchema.Required = mcpTool.InputSchema.Required

	// Convert properties
	if len(mcpTool.InputSchema.Properties) > 0 {
		gaiTool.InputSchema.Properties = make(map[string]gai.Property)
		for propName, propValue := range mcpTool.InputSchema.Properties {
			propMap, ok := propValue.(map[string]interface{})
			if !ok {
				return gai.Tool{}, fmt.Errorf("property '%s' is not a valid object", propName)
			}

			prop, err := convertProperty(propMap)
			if err != nil {
				return gai.Tool{}, fmt.Errorf("failed to convert property '%s': %w", propName, err)
			}

			gaiTool.InputSchema.Properties[propName] = prop
		}
	}

	return gaiTool, nil
}

// convertProperty converts a single property from MCP format to GAI format
func convertProperty(propMap map[string]interface{}) (gai.Property, error) {
	// Check for anyOf first - this has priority over "type"
	if anyOf, ok := propMap["anyOf"].([]interface{}); ok && len(anyOf) > 0 {
		property := gai.Property{
			Description: "", // Will be updated later if a description exists
		}

		// Handle anyOf field
		properties := make([]gai.Property, len(anyOf))
		for i, typeOption := range anyOf {
			// Handle different formats of the anyOf options
			switch typeDef := typeOption.(type) {
			case map[string]interface{}:
				// Standard object format, convert recursively
				typeProp, err := convertProperty(typeDef)
				if err != nil {
					return gai.Property{}, fmt.Errorf("failed to convert anyOf option: %w", err)
				}
				properties[i] = typeProp

			case map[string]string:
				// Simple map with just a type field
				if typeVal, ok := typeDef["type"]; ok {
					properties[i] = gai.Property{Type: stringToGAIType(typeVal)}
				} else {
					return gai.Property{}, fmt.Errorf("anyOf option map does not contain type: %v", typeDef)
				}

			case string:
				// Just a string type name
				properties[i] = gai.Property{Type: stringToGAIType(typeDef)}

			default:
				return gai.Property{}, fmt.Errorf("anyOf option has unsupported format: %T %v", typeOption, typeOption)
			}
		}

		property.AnyOf = properties

		// Add description if present
		if descStr, ok := propMap["description"].(string); ok {
			property.Description = descStr
		}

		return property, nil
	}

	// If no anyOf, proceed with normal type-based conversion
	// Extract the type
	typeStr, ok := propMap["type"].(string)
	if !ok {
		return gai.Property{}, fmt.Errorf("property type is missing or not a string")
	}

	// Create a new property
	property := gai.Property{}

	// Convert the property type
	property.Type = stringToGAIType(typeStr)
	if property.Type == gai.Null && typeStr != "null" {
		return gai.Property{}, fmt.Errorf("unsupported property type: %s", typeStr)
	}

	// Add description if present
	if descStr, ok := propMap["description"].(string); ok {
		property.Description = descStr
	}

	// Handle enumerations for string properties
	if property.Type == gai.String && propMap["enum"] != nil {
		switch enumValues := propMap["enum"].(type) {
		case []string:
			property.Enum = make([]string, len(enumValues))
			for i, val := range enumValues {
				property.Enum[i] = val
			}
		case []interface{}:
			property.Enum = make([]string, len(enumValues))
			for i, val := range enumValues {
				strVal, isString := val.(string)
				if !isString {
					return gai.Property{}, fmt.Errorf("unexpected enum value type: %T %v", val, val)
				}
				property.Enum[i] = strVal
			}
		default:
			return gai.Property{}, fmt.Errorf("enum value is not a []string")
		}
	}

	// Handle nested objects
	if property.Type == gai.Object {
		propProperties, ok := propMap["properties"].(map[string]interface{})
		if ok {
			property.Properties = make(map[string]gai.Property)
			for subPropName, subPropValue := range propProperties {
				subPropMap, ok := subPropValue.(map[string]interface{})
				if !ok {
					return gai.Property{}, fmt.Errorf("sub-property '%s' is not a valid object", subPropName)
				}

				subProp, err := convertProperty(subPropMap)
				if err != nil {
					return gai.Property{}, fmt.Errorf("failed to convert sub-property '%s': %w", subPropName, err)
				}

				property.Properties[subPropName] = subProp
			}
		}

		// Handle required fields for objects
		if required, ok := propMap["required"]; ok {
			// Try as []string first
			if requiredStrings, ok := required.([]string); ok && len(requiredStrings) > 0 {
				property.Required = requiredStrings
			} else if requiredValues, ok := required.([]interface{}); ok && len(requiredValues) > 0 {
				// If not []string, try as []interface{} and convert to []string
				property.Required = make([]string, len(requiredValues))
				for i, val := range requiredValues {
					if strVal, ok := val.(string); ok {
						property.Required[i] = strVal
					} else {
						return gai.Property{}, fmt.Errorf("required field is not a string")
					}
				}
			}
		}
	}

	// Handle arrays
	if property.Type == gai.Array {
		itemsMap, ok := propMap["items"].(map[string]interface{})
		if !ok {
			return gai.Property{}, fmt.Errorf("array items schema is missing or invalid")
		}

		itemsProp, err := convertProperty(itemsMap)
		if err != nil {
			return gai.Property{}, fmt.Errorf("failed to convert array items: %w", err)
		}

		property.Items = &itemsProp
	}

	return property, nil
}

// stringToGAIType converts a string type representation to a gai.PropertyType
func stringToGAIType(typeStr string) gai.PropertyType {
	switch typeStr {
	case "string":
		return gai.String
	case "number":
		return gai.Number
	case "integer":
		return gai.Integer
	case "boolean":
		return gai.Boolean
	case "object":
		return gai.Object
	case "array":
		return gai.Array
	case "null":
		return gai.Null
	default:
		// Default to Null for unknown types
		return gai.Null
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

// RegisterMCPToolsWithGenerator registers all MCP tools returned by the server
// with the provided gai.ToolGenerator. It converts each MCP tool description
// (already returned in gai.Tool form) into a callback that proxies execution
// through the MCP client.
func RegisterMCPToolsWithGenerator(ctx context.Context, client *Client, toolGen *gai.ToolGenerator) error {
	tools, err := client.ListTools(ctx)
	if err != nil {
		return fmt.Errorf("failed to list MCP tools: %w", err)
	}

	for _, t := range tools {
		cb := &toolCallback{client: client, toolName: t.Name}
		if err := toolGen.Register(t, cb); err != nil {
			return fmt.Errorf("failed to register tool %s: %w", t.Name, err)
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

	ctx := context.Background()
	clientInfo := ClientInfo{Name: "example", Version: "1.0"}
	capabilities := ClientCapabilities{}

	mcpClient, err := NewClient(ctx, transport, clientInfo, capabilities, DefaultOptions())
	if err != nil {
		panic(err)
	}
	defer mcpClient.Close()

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
