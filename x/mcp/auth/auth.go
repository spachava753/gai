package auth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"golang.org/x/oauth2"
	"net/http"
	"net/url"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

const (
	wellKnownEndpoint = "/.well-known/oauth-authorization-server"
	mcpServerHeader   = "MCP-Protocol-Version"
)

type Transport struct {
	Base         http.RoundTripper
	ClientName   string
	RedirectUri  string
	RedirectPort string
}

func (a *Transport) RoundTrip(request *http.Request) (*http.Response, error) {
	resp, err := a.Base.RoundTrip(request)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusUnauthorized {
		return resp, err
	}

	// Parse server URL to get base URL
	u, err := url.Parse(request.URL.String())
	if err != nil {
		return nil, fmt.Errorf("invalid server URL: %w", err)
	}

	// Authorization base URL is the scheme + host (no path)
	baseURL := fmt.Sprintf("%s://%s", u.Scheme, u.Host)
	metadataEndpoint := baseURL + wellKnownEndpoint

	// Discover server metadata
	req, err := http.NewRequestWithContext(
		request.Context(),
		http.MethodGet,
		metadataEndpoint,
		nil,
	)
	if err != nil {
		return nil, err
	}

	req.Header.Set(mcpServerHeader, resp.Header.Get(mcpServerHeader))

	httpClient := http.Client{
		Transport: a.Base,
	}

	resp, err = httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// TODO: if metadata discovery failed, should we use defaults?
	// metadata.AuthorizationEndpoint = baseURL + "/authorize"
	// metadata.TokenEndpoint = baseURL + "/token"
	// metadata.RegistrationEndpoint = baseURL + "/register"
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("metadata discovery failed: HTTP %d", resp.StatusCode)
	}

	var metadata ServerMetadata
	if err = json.NewDecoder(resp.Body).Decode(&metadata); err != nil {
		return nil, fmt.Errorf("failed to decode metadata: %w", err)
	}

	registration := map[string]interface{}{
		"client_name":   a.ClientName,
		"redirect_uris": []string{a.RedirectUri},
	}

	data, err := json.Marshal(registration)
	if err != nil {
		return nil, err
	}

	req, err = http.NewRequestWithContext(
		request.Context(),
		http.MethodPost,
		metadata.RegistrationEndpoint,
		strings.NewReader(string(data)),
	)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err = httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 201 && resp.StatusCode != 200 {
		return nil, fmt.Errorf("client registration failed: HTTP %d", resp.StatusCode)
	}

	var dynamicClient DynamicClient
	if err = json.NewDecoder(resp.Body).Decode(&dynamicClient); err != nil {
		return nil, fmt.Errorf("failed to decode registration response: %w", err)
	}

	conf := &oauth2.Config{
		ClientID:     dynamicClient.ClientID,
		ClientSecret: dynamicClient.ClientSecret,
		Endpoint: oauth2.Endpoint{
			AuthURL:  metadata.AuthorizationEndpoint,
			TokenURL: metadata.TokenEndpoint,
		},
		RedirectURL: a.RedirectUri,
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

	server := &http.Server{Addr: ":" + a.RedirectPort, Handler: mux}
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
		return nil, err
	}
	if err != nil {
		fmt.Println("Failed to open browser:", err)
	}

	var authCode string
	select {
	case authCode = <-authCodeChan:
	case err := <-errorChan:
		return nil, fmt.Errorf("authorization failed: %w", err)
	case <-time.After(2 * time.Minute):
		return nil, errors.New("authorization timeout - user did not complete OAuth flow")
	}

	tok, err := conf.Exchange(request.Context(), authCode, oauth2.VerifierOption(verifier))
	if err != nil {
		return nil, fmt.Errorf("failed to exchange code for token: %w", err)
	}

	a.Base = &oauth2.Transport{
		Source: oauth2.ReuseTokenSource(nil, conf.TokenSource(request.Context(), tok)),
		Base:   a.Base,
	}

	return a.Base.RoundTrip(request)
}
