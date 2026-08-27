package gai

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// Test data structures
type SimpleParams struct {
	Message string `json:"message"`
}

type ValidatedParams struct {
	Name  string `json:"name"`
	Email string `json:"email"`
}

func (p *ValidatedParams) Validate() error {
	if p.Name == "" {
		return fmt.Errorf("name is required")
	}
	if p.Email == "" {
		return fmt.Errorf("email is required")
	}
	return nil
}

// Test callback functions
func simpleCallback(ctx context.Context, params SimpleParams) (string, error) {
	return fmt.Sprintf("Received: %s", params.Message), nil
}

func validatedCallback(ctx context.Context, params ValidatedParams) (string, error) {
	return fmt.Sprintf("User: %s (%s)", params.Name, params.Email), nil
}

func errorCallback(ctx context.Context, params SimpleParams) (string, error) {
	return "", fmt.Errorf("deliberate error")
}

func TestCallbackDispatchScenarios(t *testing.T) {
	t.Run("ToolCallBackFunc/Call", testToolCallBackFunc_Call)
	t.Run("ToolCallBackFunc/DirectExecution", testToolCallBackFunc_DirectExecution)
}

func testToolCallBackFunc_Call(t *testing.T) {
	tests := []struct {
		name        string
		callback    ToolCallback
		parameters  map[string]any
		want        string
		wantErr     bool
		errContains string
	}{
		{
			name:       "simple successful callback",
			callback:   ToolCallBackFunc[SimpleParams](simpleCallback),
			parameters: map[string]any{"message": "Hello, world!"},
			want:       "Received: Hello, world!",
			wantErr:    false,
		},
		{
			name:       "validated parameters success",
			callback:   ToolCallBackFunc[ValidatedParams](validatedCallback),
			parameters: map[string]any{"name": "John Doe", "email": "john@example.com"},
			want:       "User: John Doe (john@example.com)",
			wantErr:    false,
		},
		{
			name:       "validation failure",
			callback:   ToolCallBackFunc[ValidatedParams](validatedCallback),
			parameters: map[string]any{"name": ""},
			want:       "parameter validation failed: name is required",
			wantErr:    false,
		},
		{
			name:        "unmarshal error",
			callback:    ToolCallBackFunc[SimpleParams](simpleCallback),
			parameters:  map[string]any{"message": 123}, // Type mismatch should cause unmarshal error
			want:        "",
			wantErr:     true,
			errContains: "failed to unmarshal",
		},
		{
			name:       "callback error",
			callback:   ToolCallBackFunc[SimpleParams](errorCallback),
			parameters: map[string]any{"message": "Hello"},
			want:       "deliberate error",
			wantErr:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			message, err := tt.callback.Call(context.Background(), tt.parameters)

			// Check error
			if (err != nil) != tt.wantErr {
				t.Errorf("Expected error: %v, got error: %v", tt.wantErr, err != nil)
				return
			}

			if err != nil {
				if tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("Error does not contain expected substring.\nExpected substring: %q\nGot error: %q",
						tt.errContains, err.Error())
				}
				return
			}

			// Verify the message
			if message.Role != ToolResult {
				t.Errorf("Expected message role ToolResult, got %v", message.Role)
			}

			if len(message.Blocks) != 1 {
				t.Errorf("Expected 1 block, got %d", len(message.Blocks))
				return
			}

			block := message.Blocks[0]

			if block.BlockType != Content {
				t.Errorf("Expected BlockType Content, got %q", block.BlockType)
			}

			if block.ModalityType != Text {
				t.Errorf("Expected ModalityType Text, got %v", block.ModalityType)
			}

			if block.Content.String() != tt.want {
				t.Errorf("Expected content %q, got %q", tt.want, block.Content.String())
			}
		})
	}
}

// TestToolCallBackFunc_DirectExecution tests that ToolCallBackFunc creates
// tool-result messages when called with decoded tool parameters.
func testToolCallBackFunc_DirectExecution(t *testing.T) {
	// Create a tool and its parameter type
	type GreetParams struct {
		Name string `json:"name"`
	}

	// Call the callback directly, as an application-owned tool loop would do.
	toolCall := ToolCallBackFunc[GreetParams](func(ctx context.Context, params GreetParams) (string, error) {
		return fmt.Sprintf("Hello, %s!", params.Name), nil
	})

	// Call the function directly with decoded parameters
	result, err := toolCall.Call(
		context.Background(),
		map[string]any{"name": "World"},
	)

	if err != nil {
		t.Fatalf("ToolCallBackFunc failed: %v", err)
	}

	// Verify result
	if result.Role != ToolResult {
		t.Errorf("Expected ToolResult role, got %v", result.Role)
	}

	if len(result.Blocks) != 1 {
		t.Fatalf("Expected 1 block, got %d", len(result.Blocks))
	}

	if result.Blocks[0].Content.String() != "Hello, World!" {
		t.Errorf("Expected 'Hello, World!', got %q", result.Blocks[0].Content.String())
	}
}
