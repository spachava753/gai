package mcp_test

import (
	"context"
	"errors"
	"fmt"
	"github.com/spachava753/gai"
	"github.com/spachava753/gai/mcp"
	"golang.org/x/oauth2"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"time"
)

func ExampleNewStdio() {
	wd, err := os.Getwd()
	if err != nil {
		fmt.Println(err)
		return
	}
	config := mcp.StdioConfig{
		Command: "docker",
		Args:    []string{"run", "-i", "--mount", fmt.Sprintf("type=bind,src=%s,dst=/projects/workspace", wd), "desktopcommander", "/projects/workspace"},
	}
	transport := mcp.NewStdio(config)
	ctx := context.Background()
	clientInfo := mcp.ClientInfo{Name: "test-client", Version: "1.0.0"}
	capabilities := mcp.ClientCapabilities{}

	client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err != nil {
		fmt.Println("Failed to create client:", err)
		return
	}
	defer client.Close()

	fmt.Println("Connected:", client.IsConnected())
	serverInfo := client.GetServerInfo()
	fmt.Println("Server name:", serverInfo.Name)
	tools, err := client.ListTools(ctx)
	if err != nil {
		fmt.Println("ListTools error:", err)
		return
	}
	if len(tools) == 0 {
		fmt.Println("No tools returned by server")
		return
	}
	// Find list_allowed_directories tool and invoke it
	for _, tool := range tools {
		if tool.Name == "list_directory" {
			msg, err := client.CallTool(ctx, tool.Name, map[string]interface{}{"path": "."})
			_ = msg
			if err != nil {
				fmt.Println("Error calling list_directory:", err)
			} else {
				fmt.Println("Got list_directory")
			}
			break
		}
	}
	if err := client.Ping(ctx); err != nil {
		fmt.Println("Ping failed:", err)
	} else {
		fmt.Println("Ping succeeded")
	}
	// Output: Connected: true
	// Server name: desktop-commander
	// Got list_directory
	// Ping succeeded
}

func ExampleNewHTTPSSE() {
	config := mcp.HTTPConfig{
		URL: "https://docs.mcp.cloudflare.com/sse",
	}
	transport := mcp.NewHTTPSSE(config)
	ctx := context.Background()
	clientInfo := mcp.ClientInfo{
		Name:    "test-client-sse",
		Version: "1.0.0",
	}
	capabilities := mcp.ClientCapabilities{}

	client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err != nil {
		fmt.Println("Failed to create HTTPSSE client:", err)
		return
	}
	defer client.Close()

	prompts, err := client.ListPrompts(ctx)
	if err != nil {
		fmt.Println("ListPrompts error:", err)
		return
	}
	fmt.Printf("Fetched %d prompt(s)\n", len(prompts))

	tools, err := client.ListTools(ctx)
	if err != nil {
		fmt.Println("ListTools error:", err)
		return
	}
	var searchTool *gai.Tool
	for i := range tools {
		if tools[i].Name == "search_cloudflare_documentation" {
			searchTool = &tools[i]
			break
		}
	}
	if searchTool != nil {
		args := map[string]interface{}{"query": "what is a worker?"}
		result, err := client.CallTool(ctx, searchTool.Name, args)
		if err != nil {
			fmt.Println("Tool call failed:", err)
		} else {
			fmt.Printf("Tool returned %d block(s)\n", len(result.Blocks))
		}
	}
	// Output: Fetched 1 prompt(s)
	// Tool returned 1 block(s)
}

func ExampleNewStreamableHTTP() {
	config := mcp.HTTPConfig{
		URL: "https://mcp.deepwiki.com/mcp",
	}
	transport := mcp.NewStreamableHTTP(config)
	ctx := context.Background()
	clientInfo := mcp.ClientInfo{
		Name:    "test-client-sse",
		Version: "1.0.0",
	}
	capabilities := mcp.ClientCapabilities{}

	client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err != nil {
		fmt.Println("Failed to create StreamableHTTP client:", err)
		return
	}
	defer client.Close()
	tools, err := client.ListTools(ctx)
	if err != nil {
		fmt.Println("ListTools error:", err)
		return
	}
	var toolName string
	for i := range tools {
		if tools[i].Name == "read_wiki_structure" {
			toolName = tools[i].Name
			break
		}
	}
	if toolName != "" {
		callCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
		defer cancel()
		result, err := client.CallTool(callCtx, toolName, map[string]interface{}{
			"repoName": "spachava753/gai",
		})
		if err != nil {
			fmt.Println("Tool call failed:", err)
			return
		}
		fmt.Printf("read_wiki_structure blocks: %d\n", len(result.Blocks))
	}
	// Output: read_wiki_structure blocks: 1
}

func ExampleDynamicRegistration() {
	ctx := context.Background()

	serverUrl := "https://ai-gateway.mcp.cloudflare.com/sse"

	// We expect this to fail
	config := mcp.HTTPConfig{
		URL:        serverUrl,
		HTTPClient: http.DefaultClient,
	}

	transport := mcp.NewHTTPSSE(config)

	clientInfo := mcp.ClientInfo{
		Name:    "test-client-auth-code-grant",
		Version: "1.0.0",
	}
	capabilities := mcp.ClientCapabilities{}

	client, err := mcp.NewClient(ctx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err == nil || !errors.Is(err, mcp.AuthenticationError) {
		fmt.Println("expected to fail with authentication error")
		return
	}

	// Perform full OAuth Authorization Code Grant flow
	fmt.Println("Starting OAuth Authorization Code Grant flow with PKCE")

	redirectPort := "18456"
	redirectURI := "http://localhost:" + redirectPort + "/callback"

	// Discover server metadata
	fmt.Println("Discovering OAuth server metadata...")
	metadataEndpoint, err := mcp.DefaultMetadataEndpoint(serverUrl)
	if err != nil {
		fmt.Println("Failed to create metadata endpoint:", err)
		return
	}
	metadata, err := mcp.DiscoverServerMetadata(ctx, metadataEndpoint, mcp.ProtocolVersion, http.DefaultClient)
	if err != nil {
		fmt.Println("Metadata discovery failed:", err)
		fmt.Println("Using default OAuth endpoints")
		metadata, err = mcp.FallbackServerMetadata(serverUrl)
		if err != nil {
			fmt.Println("Failed to discover server metadata:", err)

			return
		}
	} else {
		fmt.Println("Successfully discovered server metadata")
		fmt.Println("  Authorization:", metadata.AuthorizationEndpoint)
		fmt.Println("  Token:", metadata.TokenEndpoint)
		if metadata.RegistrationEndpoint != "" {
			fmt.Println("  Registration:", metadata.RegistrationEndpoint)
		}
	}

	fmt.Println("Attempting dynamic client registration...")
	dynamicClient, err := mcp.DynamicRegistration(
		ctx,
		metadata.RegistrationEndpoint,
		[]string{redirectURI},
		"mcp-test-client",
		http.DefaultClient,
	)
	if err != nil {
		fmt.Println("Dynamic client registration failed:", err)
		return
	}

	fmt.Println("Successfully registered client")

	conf := &oauth2.Config{
		ClientID:     dynamicClient.ClientID,
		ClientSecret: dynamicClient.ClientSecret,
		Endpoint: oauth2.Endpoint{
			AuthURL:  metadata.AuthorizationEndpoint,
			TokenURL: metadata.TokenEndpoint,
		},
		RedirectURL: redirectURI,
	}

	s := "state"
	verifier := oauth2.GenerateVerifier()

	authCodeURL := conf.AuthCodeURL(
		s,
		oauth2.AccessTypeOffline,
		oauth2.S256ChallengeOption(verifier),
	)

	authCodeChan := make(chan string, 1)
	errorChan := make(chan error, 1)

	mux := http.NewServeMux()
	mux.HandleFunc("/callback", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("state") != s {
			errorChan <- fmt.Errorf("state mismatch: expected %s, got %s", s, r.URL.Query().Get("state"))
			http.Error(w, "State mismatch", http.StatusBadRequest)
			return
		}
		if errParam := r.URL.Query().Get("error"); errParam != "" {
			errorChan <- fmt.Errorf("OAuth error: %s - %s", errParam, r.URL.Query().Get("error_description"))
			http.Error(w, "Authorization failed: "+errParam, http.StatusBadRequest)
			return
		}
		code := r.URL.Query().Get("code")
		if code == "" {
			errorChan <- fmt.Errorf("no authorization code in callback")
			http.Error(w, "No authorization code", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprintf(w, "<!DOCTYPE html><html><head><title>Authorization Successful</title></head><body><h1>Authorization Successful!</h1><p>You can close this window and return to the test.</p></body></html>")
		authCodeChan <- code
	})

	server := &http.Server{Addr: ":" + redirectPort, Handler: mux}
	go func() {
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errorChan <- fmt.Errorf("callback server error: %v", err)
		}
	}()
	defer server.Shutdown(context.Background())

	switch runtime.GOOS {
	case "linux":
		err = exec.Command("xdg-open", authCodeURL).Start()
	case "windows":
		err = exec.Command("rundll32.exe", "url.dll,FileProtocolHandler", authCodeURL).Start()
	case "darwin": // macOS
		err = exec.Command("open", authCodeURL).Start()
	default:
		err = fmt.Errorf("unsupported platform: %s", runtime.GOOS)
		return
	}
	if err != nil {
		fmt.Println("Failed to open browser:", err)
		return
	}

	var authCode string
	select {
	case authCode = <-authCodeChan:
		fmt.Println("Received authorization code")
	case err := <-errorChan:
		fmt.Println("Authorization failed:", err)
		return
	case <-time.After(2 * time.Minute):
		fmt.Println("Authorization timeout - user did not complete OAuth flow.")
		return
	}

	fmt.Println("Exchanging authorization code for access token...")
	tok, err := conf.Exchange(ctx, authCode, oauth2.VerifierOption(verifier))
	if err != nil {
		fmt.Println("Failed to exchange code for token:", err)
		return
	}

	fmt.Println("Successfully obtained access token")
	if tok.ExpiresIn > 0 {
		fmt.Printf("Token expires in %d seconds\n", tok.ExpiresIn)
	}

	config = mcp.HTTPConfig{
		URL:        serverUrl,
		HTTPClient: conf.Client(context.WithValue(context.Background(), oauth2.HTTPClient, http.DefaultClient), tok),
	}
	transport = mcp.NewHTTPSSE(config)

	newClientCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	client, err = mcp.NewClient(newClientCtx, transport, clientInfo, capabilities, mcp.DefaultOptions())
	if err != nil {
		fmt.Println("Failed to create client with OAuth token:", err)
		return
	}
	defer client.Close()

	listToolsCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	tools, err := client.ListTools(listToolsCtx)
	if err != nil {
		fmt.Println("Failed to list tools:", err)
		return
	}

	if len(tools) == 0 {
		fmt.Println("Expected tools list to be non-empty")
		return
	}

	fmt.Printf("Successfully fetched %d tools via OAuth authenticated connection\n", len(tools))
	for i, tool := range tools {
		fmt.Printf("  %d. %s\n", i+1, tool.Name)
	}
	// Output: Starting OAuth Authorization Code Grant flow with PKCE
	// Discovering OAuth server metadata...
	// Successfully discovered server metadata
	//   Authorization: https://ai-gateway.mcp.cloudflare.com/oauth/authorize
	//   Token: https://ai-gateway.mcp.cloudflare.com/token
	//   Registration: https://ai-gateway.mcp.cloudflare.com/register
	// Attempting dynamic client registration...
	// Successfully registered client
	// Received authorization code
	// Exchanging authorization code for access token...
	// Successfully obtained access token
	// Token expires in 3600 seconds
	// Successfully fetched 7 tools via OAuth authenticated connection
	//   1. accounts_list
	//   2. set_active_account
	//   3. list_gateways
	//   4. list_logs
	//   5. get_log_details
	//   6. get_log_request_body
	//   7. get_log_response_body
}
