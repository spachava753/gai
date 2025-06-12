package mcp_test

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/spachava753/gai"
	"github.com/spachava753/gai/mcp"
)

func TestClient_BasicFlow(t *testing.T) {
	// This test requires a real MCP server running
	// Skip if not in integration test mode
	if testing.Short() {
		t.Skip("Skipping integration test")
	}

	// Create stdio transport
	config := mcp.StdioConfig{
		Command: "docker",
		Args:    []string{"run", "-i", "--rm", "mcp/time"},
	}

	transport := mcp.NewStdio(config)

	ctx := context.Background()
	clientInfo := mcp.ClientInfo{
		Name:    "test-client",
		Version: "1.0.0",
	}
	capabilities := mcp.ClientCapabilities{}

	client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	if !client.IsConnected() {
		t.Error("Client should be connected")
	}

	// Test server info
	serverInfo := client.GetServerInfo()
	if serverInfo.Name == "" {
		t.Error("Server info should have a name")
	}

	// Test tools
	toolList, err := client.ListTools(ctx)
	if err != nil {
		t.Fatalf("Failed to list tools: %v", err)
	}

	if len(toolList) == 0 {
		t.Error("Expected at least one tool")
	}

	// Test tool call
	for _, tool := range toolList {
		if tool.Name == "get_current_time" {
			result, err := client.CallTool(ctx, tool.Name, map[string]interface{}{
				"timezone": "UTC",
			})
			if err != nil {
				t.Errorf("Failed to call tool %s: %v", tool.Name, err)
			} else {
				t.Logf("Tool %s result: %v", tool.Name, result)
			}
			break
		}
	}

	// Test ping
	t.Log("testing ping")
	if err = client.Ping(ctx); err != nil {
		t.Errorf("Ping failed: %v", err)
	}
}

func TestClient_ErrorHandling(t *testing.T) {
	// Test connection errors
	config := mcp.StdioConfig{
		Command: "nonexistent-command",
		Args:    []string{},
	}

	transport := mcp.NewStdio(config)

	ctx := context.Background()
	clientInfo := mcp.ClientInfo{
		Name:    "test-client",
		Version: "1.0.0",
	}
	capabilities := mcp.ClientCapabilities{}

	// Should fail to create client due to connection error
	client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err == nil {
		defer client.Close()
		t.Error("Expected connection error")
	}

	// The error should be related to the command not being found
	if err != nil && !strings.Contains(err.Error(), "failed to connect") {
		t.Errorf("Expected connection error, got: %v", err)
	}
}

func TestClient_HTTPSSE_Integration(t *testing.T) {
	// Create HTTP transport for the public MCP server
	config := mcp.HTTPConfig{
		URL: "https://docs.mcp.cloudflare.com/sse",
	}

	transport := mcp.NewHTTP(config)

	ctx := context.Background()
	clientInfo := mcp.ClientInfo{
		Name:    "test-client-sse",
		Version: "1.0.0",
	}
	capabilities := mcp.ClientCapabilities{}

	client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// 1. Fetch prompts
	prompts, err := client.ListPrompts(ctx)
	if err != nil {
		t.Fatalf("Failed to list prompts: %v", err)
	}
	if len(prompts) == 0 {
		t.Fatal("Expected prompts list to be non-empty, but it was empty")
	}
	t.Logf("Successfully fetched %d prompts.", len(prompts))

	// 2. Fetch tools
	tools, err := client.ListTools(ctx)
	if err != nil {
		t.Fatalf("Failed to list tools: %v", err)
	}
	if len(tools) == 0 {
		t.Fatal("Expected tools list to be non-empty, but it was empty")
	}
	t.Logf("Successfully fetched %d tools.", len(tools))

	// Find the search tool
	var searchTool *gai.Tool
	for i, tool := range tools {
		if tool.Name == "search_cloudflare_documentation" {
			searchTool = &tools[i]
			break
		}
	}

	if searchTool == nil {
		t.Fatal("Could not find the 'search' tool in the tool list")
	}

	// 3. Call the search tool
	t.Logf("Calling tool: %s", searchTool.Name)
	args := map[string]interface{}{
		"query": "what is a worker?",
	}
	result, err := client.CallTool(ctx, searchTool.Name, args)
	if err != nil {
		t.Fatalf("Tool call failed: %v", err)
	}

	if result == nil {
		t.Fatal("Tool call result was nil")
	}

	t.Logf("Successfully called tool '%s' with query '%s'.", searchTool.Name, args["query"])
}

func TestClient_StreamableHTTP_Integration(t *testing.T) {
	// Create HTTP transport for the public MCP server
	config := mcp.HTTPConfig{
		URL: "https://mcp.deepwiki.com/mcp",
	}

	transport := mcp.NewHTTP(config)

	ctx := context.Background()
	clientInfo := mcp.ClientInfo{
		Name:    "test-client-sse",
		Version: "1.0.0",
	}
	capabilities := mcp.ClientCapabilities{}

	client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Fetch tools
	tools, err := client.ListTools(ctx)
	if err != nil {
		t.Fatalf("Failed to list tools: %v", err)
	}
	if len(tools) == 0 {
		t.Fatal("Expected tools list to be non-empty, but it was empty")
	}
	t.Logf("Successfully fetched %d tools.", len(tools))

	// Find the search tool
	var askQuestionTool *gai.Tool
	for i, tool := range tools {
		if tool.Name == "ask_question" {
			askQuestionTool = &tools[i]
			break
		}
	}

	if askQuestionTool == nil {
		t.Fatal("Could not find the 'search' tool in the tool list")
	}

	// 3. Call the search tool
	t.Logf("Calling tool: %s", askQuestionTool.Name)
	args := map[string]interface{}{
		"question": "what does the generator interface do?",
		"repoName": "spachava753/gai",
	}
	result, err := client.CallTool(ctx, askQuestionTool.Name, args)
	if err != nil {
		t.Fatalf("Tool call failed: %v", err)
	}

	if result == nil {
		t.Fatal("Tool call result was nil")
	}

	t.Logf("Successfully called tool '%s' with question '%s' on repo %s.", askQuestionTool.Name, args["question"], args["repoName"])
}

func TestClient_HTTPSSE_AuthCodeGrant_Integration(t *testing.T) {
	// This test connects to an authenticated MCP server using OAuth Authorization Code Grant with PKCE.
	// It demonstrates the full OAuth flow including:
	//   1. Server metadata discovery
	//   2. Optional dynamic client registration
	//   3. Authorization Code Grant with PKCE
	//   4. Token exchange
	//   5. Authenticated MCP requests
	//
	// To run this test:
	//   Option 1: Set MCP_TEST_ACCESS_TOKEN to skip OAuth and use existing token
	//   Option 2: Set MCP_TEST_CLIENT_ID to use pre-registered client
	//   Option 3: Leave empty to attempt dynamic client registration
	//
	// The test will:
	//   1. Start a local HTTP server for OAuth callback
	//   2. Generate authorization URL with PKCE
	//   3. Wait for user to authorize in browser
	//   4. Exchange authorization code for access token
	//   5. Make authenticated MCP requests

	if os.Getenv("ENABLE_MANUAL_TESTS") != "true" {
		t.Skip("Skipping test, requires manual input due to browser access")
	}

	// Check for existing credentials
	clientID := os.Getenv("MCP_TEST_CLIENT_ID")
	clientSecret := os.Getenv("MCP_TEST_CLIENT_SECRET")
	accessToken := os.Getenv("MCP_TEST_ACCESS_TOKEN")

	serverURL := "https://ai-gateway.mcp.cloudflare.com/sse"

	// If we already have an access token, use it directly
	if accessToken != "" {
		t.Log("Using provided access token, skipping OAuth flow")

		config := mcp.HTTPConfig{
			URL: serverURL,
			Headers: map[string]string{
				"Authorization": "Bearer " + accessToken,
			},
		}

		transport := mcp.NewHTTP(config)

		ctx := context.Background()
		clientInfo := mcp.ClientInfo{
			Name:    "test-client-auth-code-grant",
			Version: "1.0.0",
		}
		capabilities := mcp.ClientCapabilities{}

		client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, mcp.DefaultOptions())
		if err != nil {
			t.Fatalf("Failed to create client: %v", err)
		}
		defer client.Close()

		// List tools
		tools, err := client.ListTools(ctx)
		if err != nil {
			t.Fatalf("Failed to list tools: %v", err)
		}

		if len(tools) == 0 {
			t.Fatal("Expected tools list to be non-empty")
		}

		t.Logf("Successfully fetched %d tools", len(tools))
		return
	}

	// Perform full OAuth Authorization Code Grant flow
	t.Log("Starting OAuth Authorization Code Grant flow with PKCE")

	ctx := context.Background()
	redirectPort := "18456"
	redirectURI := "http://localhost:" + redirectPort + "/callback"

	// Create auth manager for metadata discovery
	authConfig := mcp.AuthConfig{
		ClientID:     clientID,
		ClientSecret: clientSecret,
		RedirectURI:  redirectURI,
	}

	authManager := mcp.NewAuthManager(authConfig)

	// Discover server metadata
	t.Log("Discovering OAuth server metadata...")
	metadata, err := authManager.DiscoverServerMetadata(ctx, serverURL)
	if err != nil {
		t.Logf("Metadata discovery failed: %v", err)
		t.Log("Using default OAuth endpoints")

		// Use default endpoints
		baseURL, _ := getBaseURL(serverURL)
		authConfig.AuthorizationEndpoint = baseURL + "/authorize"
		authConfig.TokenEndpoint = baseURL + "/token"
		authConfig.RegistrationEndpoint = baseURL + "/register"
		authManager = mcp.NewAuthManager(authConfig)
	} else {
		t.Log("Successfully discovered server metadata")
		authConfig.AuthorizationEndpoint = metadata.AuthorizationEndpoint
		authConfig.TokenEndpoint = metadata.TokenEndpoint
		authConfig.RegistrationEndpoint = metadata.RegistrationEndpoint
		authManager = mcp.NewAuthManager(authConfig)

		// Log discovered endpoints
		t.Logf("  Authorization: %s", metadata.AuthorizationEndpoint)
		t.Logf("  Token: %s", metadata.TokenEndpoint)
		if metadata.RegistrationEndpoint != "" {
			t.Logf("  Registration: %s", metadata.RegistrationEndpoint)
		}
	}

	// Dynamic client registration if no client ID
	if clientID == "" {
		if authConfig.RegistrationEndpoint == "" {
			t.Skip("No client ID provided and dynamic registration not available")
		}

		t.Log("Attempting dynamic client registration...")
		registration, err := authManager.RegisterClient(ctx, authConfig.RegistrationEndpoint, "mcp-test-client")
		if err != nil {
			t.Fatalf("Dynamic client registration failed: %v", err)
		}

		t.Logf("Successfully registered client: %s", registration.ClientID)
		clientID = registration.ClientID
		clientSecret = registration.ClientSecret

		// Update auth manager with new credentials
		authConfig.ClientID = clientID
		authConfig.ClientSecret = clientSecret
		authManager = mcp.NewAuthManager(authConfig)
	}

	// Generate PKCE parameters
	codeVerifier, codeChallenge, err := mcp.GeneratePKCE()
	if err != nil {
		t.Fatalf("Failed to generate PKCE: %v", err)
	}
	t.Log("Generated PKCE challenge")

	// Generate state for CSRF protection
	state := generateRandomState()

	// Generate authorization URL
	authURL, err := authManager.GetAuthorizationURL(state, codeChallenge)
	if err != nil {
		t.Fatalf("Failed to generate authorization URL: %v", err)
	}

	// Set up OAuth callback server
	authCodeChan := make(chan string, 1)
	errorChan := make(chan error, 1)

	mux := http.NewServeMux()
	mux.HandleFunc("/callback", func(w http.ResponseWriter, r *http.Request) {
		// Validate state
		if r.URL.Query().Get("state") != state {
			err := fmt.Errorf("state mismatch: expected %s, got %s", state, r.URL.Query().Get("state"))
			errorChan <- err
			http.Error(w, "State mismatch", http.StatusBadRequest)
			return
		}

		// Check for OAuth error
		if errParam := r.URL.Query().Get("error"); errParam != "" {
			err := fmt.Errorf("OAuth error: %s - %s",
				errParam,
				r.URL.Query().Get("error_description"))
			errorChan <- err
			http.Error(w, "Authorization failed: "+errParam, http.StatusBadRequest)
			return
		}

		// Get authorization code
		code := r.URL.Query().Get("code")
		if code == "" {
			err := fmt.Errorf("no authorization code in callback")
			errorChan <- err
			http.Error(w, "No authorization code", http.StatusBadRequest)
			return
		}

		// Send success response to browser
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprintf(w, `<!DOCTYPE html>
<html>
<head>
    <title>Authorization Successful</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        .success { color: green; }
    </style>
</head>
<body>
    <h1 class="success">Authorization Successful!</h1>
    <p>You can close this window and return to the test.</p>
    <script>
        // Try to close the window
        setTimeout(function() { window.close(); }, 2000);
    </script>
</body>
</html>`)

		// Send code through channel
		authCodeChan <- code
	})

	server := &http.Server{
		Addr:    ":" + redirectPort,
		Handler: mux,
	}

	// Start callback server
	go func() {
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errorChan <- fmt.Errorf("callback server error: %v", err)
		}
	}()
	defer server.Shutdown(context.Background())

	// Display authorization instructions
	t.Log("=== OAuth Authorization Required ===")
	t.Logf("Please visit the following URL to authorize:")
	t.Logf("")
	t.Logf("  %s", authURL)
	t.Logf("")
	t.Log("Waiting for authorization (timeout: 2 minutes)...")

	// For automated testing, check if we should open browser
	if os.Getenv("MCP_TEST_OPEN_BROWSER") == "true" {
		t.Log("Note: Automatic browser opening not implemented")
	}

	// Wait for authorization with timeout
	var authCode string
	select {
	case authCode = <-authCodeChan:
		t.Log("Received authorization code")
	case err := <-errorChan:
		t.Fatalf("Authorization failed: %v", err)
	case <-time.After(2 * time.Minute):
		t.Skip("Authorization timeout - user did not complete OAuth flow. " +
			"Set MCP_TEST_ACCESS_TOKEN to skip OAuth flow.")
	}

	// Exchange authorization code for access token
	t.Log("Exchanging authorization code for access token...")
	tokenResp, err := authManager.ExchangeCode(ctx, authCode, codeVerifier)
	if err != nil {
		t.Fatalf("Failed to exchange code for token: %v", err)
	}

	t.Log("Successfully obtained access token")
	if tokenResp.ExpiresIn > 0 {
		t.Logf("Token expires in %d seconds", tokenResp.ExpiresIn)
	}

	// Now use the access token with MCP
	config := mcp.HTTPConfig{
		URL: serverURL,
		Headers: map[string]string{
			"Authorization": "Bearer " + tokenResp.AccessToken,
		},
	}

	transport := mcp.NewHTTP(config)

	clientInfo := mcp.ClientInfo{
		Name:    "test-client-auth-code-grant",
		Version: "1.0.0",
	}
	capabilities := mcp.ClientCapabilities{}

	client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err != nil {
		t.Fatalf("Failed to create client with OAuth token: %v", err)
	}
	defer client.Close()

	// List tools to verify authentication works
	tools, err := client.ListTools(ctx)
	if err != nil {
		t.Fatalf("Failed to list tools: %v", err)
	}

	if len(tools) == 0 {
		t.Fatal("Expected tools list to be non-empty")
	}

	t.Logf("Successfully fetched %d tools via OAuth authenticated connection", len(tools))

	// Display tool names
	for i, tool := range tools {
		t.Logf("  %d. %s", i+1, tool.Name)
	}

	// Optionally display the access token for manual testing
	if os.Getenv("MCP_TEST_SHOW_TOKEN") == "true" {
		t.Logf("Access token for reuse: %s", tokenResp.AccessToken)
	}
}

// Helper function to generate random state for CSRF protection
func generateRandomState() string {
	b := make([]byte, 16)
	rand.Read(b)
	return base64.URLEncoding.EncodeToString(b)
}

// Helper function to get base URL from a full URL
func getBaseURL(fullURL string) (string, error) {
	u, err := url.Parse(fullURL)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s://%s", u.Scheme, u.Host), nil
}
