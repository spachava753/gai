package mcp

import (
	"context"
	"strings"
	"testing"
)

func TestCapabilityValidation(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name               string
		serverCapabilities ServerCapabilities
		testFunc           func(client *Client) error
		expectError        bool
		errorContains      string
	}{
		{
			name:               "ListTools without tools capability",
			serverCapabilities: ServerCapabilities{},
			testFunc: func(c *Client) error {
				_, err := c.ListTools(ctx)
				return err
			},
			expectError:   true,
			errorContains: "server does not advertise tools capability",
		},
		{
			name: "ListTools with tools capability passes capability check",
			serverCapabilities: ServerCapabilities{
				Tools: &Capability{},
			},
			testFunc: func(c *Client) error {
				_, err := c.ListTools(ctx)
				return err
			},
			expectError:   true, // Will still error due to not connected, but different error
			errorContains: "not connected",
		},
		{
			name:               "CallTool without tools capability",
			serverCapabilities: ServerCapabilities{},
			testFunc: func(c *Client) error {
				_, err := c.CallTool(ctx, "test", map[string]any{})
				return err
			},
			expectError:   true,
			errorContains: "server does not advertise tools capability",
		},
		{
			name:               "ListResources without resources capability",
			serverCapabilities: ServerCapabilities{},
			testFunc: func(c *Client) error {
				_, err := c.ListResources(ctx)
				return err
			},
			expectError:   true,
			errorContains: "server does not advertise resources capability",
		},
		{
			name:               "ReadResource without resources capability",
			serverCapabilities: ServerCapabilities{},
			testFunc: func(c *Client) error {
				_, err := c.ReadResource(ctx, "test://uri")
				return err
			},
			expectError:   true,
			errorContains: "server does not advertise resources capability",
		},
		{
			name:               "SubscribeToResource without resources capability",
			serverCapabilities: ServerCapabilities{},
			testFunc: func(c *Client) error {
				return c.SubscribeToResource(ctx, "test://uri")
			},
			expectError:   true,
			errorContains: "server does not support this feature",
		},
		{
			name: "SubscribeToResource without subscribe sub-capability",
			serverCapabilities: ServerCapabilities{
				Resources: &Capability{Subscribe: false},
			},
			testFunc: func(c *Client) error {
				return c.SubscribeToResource(ctx, "test://uri")
			},
			expectError:   true,
			errorContains: "server does not support this feature",
		},
		{
			name: "SubscribeToResource with subscribe sub-capability passes capability check",
			serverCapabilities: ServerCapabilities{
				Resources: &Capability{Subscribe: true},
			},
			testFunc: func(c *Client) error {
				return c.SubscribeToResource(ctx, "test://uri")
			},
			expectError:   true, // Will still error due to not connected
			errorContains: "not connected",
		},
		{
			name:               "UnsubscribeFromResource without resources capability",
			serverCapabilities: ServerCapabilities{},
			testFunc: func(c *Client) error {
				return c.UnsubscribeFromResource(ctx, "test://uri")
			},
			expectError:   true,
			errorContains: "server does not advertise resources capability",
		},
		{
			name: "UnsubscribeFromResource without subscribe sub-capability",
			serverCapabilities: ServerCapabilities{
				Resources: &Capability{Subscribe: false},
			},
			testFunc: func(c *Client) error {
				return c.UnsubscribeFromResource(ctx, "test://uri")
			},
			expectError:   true,
			errorContains: "server does not support resource subscriptions",
		},
		{
			name:               "ListPrompts without prompts capability",
			serverCapabilities: ServerCapabilities{},
			testFunc: func(c *Client) error {
				_, err := c.ListPrompts(ctx)
				return err
			},
			expectError:   true,
			errorContains: "server does not advertise prompts capability",
		},
		{
			name:               "GetPrompt without prompts capability",
			serverCapabilities: ServerCapabilities{},
			testFunc: func(c *Client) error {
				_, err := c.GetPrompt(ctx, "test", map[string]string{})
				return err
			},
			expectError:   true,
			errorContains: "server does not advertise prompts capability",
		},
		{
			name:               "SetLoggingLevel without logging capability",
			serverCapabilities: ServerCapabilities{},
			testFunc: func(c *Client) error {
				return c.SetLoggingLevel(ctx, "debug")
			},
			expectError:   true,
			errorContains: "server does not advertise logging capability",
		},
		{
			name: "SetLoggingLevel with logging capability passes capability check",
			serverCapabilities: ServerCapabilities{
				Logging: &Capability{},
			},
			testFunc: func(c *Client) error {
				return c.SetLoggingLevel(ctx, "debug")
			},
			expectError:   true, // Will still error due to not connected
			errorContains: "not connected",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a minimal client with just the parts we need for capability checking
			client := &Client{
				serverCapabilities: tt.serverCapabilities,
				initialized:        true,
			}

			// Run the test function
			err := tt.testFunc(client)

			// Check error expectations
			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error containing '%s', but got nil", tt.errorContains)
				} else if tt.errorContains != "" && !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("Expected error containing '%s', but got: %v", tt.errorContains, err)
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, but got: %v", err)
				}
			}
		})
	}
}
