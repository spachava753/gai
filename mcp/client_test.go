package mcp_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/spachava753/gai"
	"github.com/spachava753/gai/mcp"

	"golang.org/x/oauth2"
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

func TestClient_HTTPSSE_DynamicClientRegistration_Integration(t *testing.T) {
	if os.Getenv("ENABLE_MANUAL_TESTS") != "true" {
		t.Skip("Skipping test, requires manual input due to browser access")
	}

	ctx := context.Background()

	serverUrl := "https://ai-gateway.mcp.cloudflare.com/sse"

	// We expect this to fail
	config := mcp.HTTPConfig{
		URL:        serverUrl,
		HTTPClient: http.DefaultClient,
	}

	transport := mcp.NewHTTP(config)

	clientInfo := mcp.ClientInfo{
		Name:    "test-client-auth-code-grant",
		Version: "1.0.0",
	}
	capabilities := mcp.ClientCapabilities{}

	client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err == nil || !errors.Is(err, mcp.AuthenticationError) {
		t.Fatal("expected to fail with authentication error")
	}

	// Perform full OAuth Authorization Code Grant flow
	t.Log("Starting OAuth Authorization Code Grant flow with PKCE")

	redirectPort := "18456"
	redirectURI := "http://localhost:" + redirectPort + "/callback"

	// Discover server metadata
	t.Log("Discovering OAuth server metadata...")
	metadataEndpoint, err := mcp.DefaultMetadataEndpoint(serverUrl)
	if err != nil {
		t.Fatalf("Failed to create metadata endpoint: %v", err)
	}
	metadata, err := mcp.DiscoverServerMetadata(ctx, metadataEndpoint, mcp.ProtocolVersion, http.DefaultClient)
	if err != nil {
		t.Logf("Metadata discovery failed: %v", err)
		t.Log("Using default OAuth endpoints")
		//
		metadata, err = mcp.FallbackServerMetadata(serverUrl)
		if err != nil {
			t.Fatalf("Failed to discover server metadata: %v", err)
		}
	} else {
		t.Log("Successfully discovered server metadata")
		t.Logf("  Authorization: %s", metadata.AuthorizationEndpoint)
		t.Logf("  Token: %s", metadata.TokenEndpoint)
		if metadata.RegistrationEndpoint != "" {
			t.Logf("  Registration: %s", metadata.RegistrationEndpoint)
		}
	}

	t.Log("Attempting dynamic client registration...")
	dynamicClient, err := mcp.DynamicRegistration(
		ctx,
		metadata.RegistrationEndpoint,
		[]string{redirectURI},
		"mcp-test-client",
		http.DefaultClient,
	)
	if err != nil {
		t.Fatalf("Dynamic client registration failed: %v", err)
	}

	t.Logf("Successfully registered client: %s", dynamicClient.ClientID)

	conf := &oauth2.Config{
		ClientID:     dynamicClient.ClientID,
		ClientSecret: dynamicClient.ClientSecret,
		Endpoint: oauth2.Endpoint{
			AuthURL:  metadata.AuthorizationEndpoint,
			TokenURL: metadata.TokenEndpoint,
		},
		RedirectURL: redirectURI,
	}

	// use PKCE to protect against CSRF attacks
	// https://www.ietf.org/archive/id/draft-ietf-oauth-security-topics-22.html#name-countermeasures-6
	verifier := oauth2.GenerateVerifier()

	// Generate authorization URL\
	state := "state"
	authCodeURL := conf.AuthCodeURL(
		state,
		oauth2.AccessTypeOffline,
		oauth2.S256ChallengeOption(verifier),
	)

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
	t.Logf("  %s", authCodeURL)
	t.Logf("")
	t.Log("Waiting for authorization (timeout: 2 minutes)...")

	// Wait for authorization with timeout
	var authCode string
	select {
	case authCode = <-authCodeChan:
		t.Log("Received authorization code")
	case err := <-errorChan:
		t.Fatalf("Authorization failed: %v", err)
	case <-time.After(2 * time.Minute):
		t.Skip("Authorization timeout - user did not complete OAuth flow.")
	}

	// Exchange authorization code for access token
	t.Log("Exchanging authorization code for access token...")
	tok, err := conf.Exchange(ctx, authCode, oauth2.VerifierOption(verifier))
	if err != nil {
		t.Fatalf("Failed to exchange code for token: %v", err)
	}

	t.Log("Successfully obtained access token")
	if tok.ExpiresIn > 0 {
		t.Logf("Token expires in %d seconds", tok.ExpiresIn)
	}

	// Now use the access token with MCP
	config = mcp.HTTPConfig{
		URL:        serverUrl,
		HTTPClient: conf.Client(context.WithValue(context.Background(), oauth2.HTTPClient, http.DefaultClient), tok),
	}

	transport = mcp.NewHTTP(config)

	newClientCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	client, err = mcp.NewClient(newClientCtx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err != nil {
		t.Fatalf("Failed to create client with OAuth token: %v", err)
	}
	defer client.Close()

	// List tools to verify authentication works
	listToolsCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	tools, err := client.ListTools(listToolsCtx)
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
}
