// This file provides OAuth authentication for MCP HTTP transport.
package mcp

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// AuthConfig contains OAuth configuration
type AuthConfig struct {
	// AuthorizationEndpoint is the OAuth authorization endpoint
	AuthorizationEndpoint string

	// TokenEndpoint is the OAuth token endpoint
	TokenEndpoint string

	// RegistrationEndpoint is the OAuth dynamic client registration endpoint
	RegistrationEndpoint string

	// ClientID is the OAuth client ID
	ClientID string

	// ClientSecret is the OAuth client secret (for confidential clients)
	ClientSecret string

	// RedirectURI is the OAuth redirect URI
	RedirectURI string

	// Scopes are the requested OAuth scopes
	Scopes []string

	// HTTPClient is the HTTP client to use
	HTTPClient *http.Client
}

// ServerMetadata represents OAuth authorization server metadata
type ServerMetadata struct {
	Issuer                            string   `json:"issuer"`
	AuthorizationEndpoint             string   `json:"authorization_endpoint"`
	TokenEndpoint                     string   `json:"token_endpoint"`
	RegistrationEndpoint              string   `json:"registration_endpoint,omitempty"`
	ResponseTypesSupported            []string `json:"response_types_supported"`
	GrantTypesSupported               []string `json:"grant_types_supported"`
	TokenEndpointAuthMethodsSupported []string `json:"token_endpoint_auth_methods_supported"`
}

// ClientRegistration represents OAuth dynamic client registration
type ClientRegistration struct {
	ClientID     string   `json:"client_id"`
	ClientSecret string   `json:"client_secret,omitempty"`
	RedirectURIs []string `json:"redirect_uris"`
	ClientName   string   `json:"client_name"`
}

// TokenResponse represents an OAuth token response
type TokenResponse struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	ExpiresIn    int    `json:"expires_in,omitempty"`
	RefreshToken string `json:"refresh_token,omitempty"`
	Scope        string `json:"scope,omitempty"`
}

// AuthManager handles OAuth authentication for HTTP transport
type AuthManager struct {
	config      AuthConfig
	client      *http.Client
	accessToken string
	tokenExpiry time.Time
}

// NewAuthManager creates a new auth manager
func NewAuthManager(config AuthConfig) *AuthManager {
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{
			Timeout: 30 * time.Second,
		}
	}

	return &AuthManager{
		config: config,
		client: client,
	}
}

// DiscoverServerMetadata discovers OAuth server metadata
func (am *AuthManager) DiscoverServerMetadata(ctx context.Context, serverURL string) (*ServerMetadata, error) {
	// Parse server URL to get base URL
	u, err := url.Parse(serverURL)
	if err != nil {
		return nil, fmt.Errorf("invalid server URL: %w", err)
	}

	// Authorization base URL is the scheme + host (no path)
	baseURL := fmt.Sprintf("%s://%s", u.Scheme, u.Host)
	metadataURL := baseURL + "/.well-known/oauth-authorization-server"

	req, err := http.NewRequestWithContext(ctx, "GET", metadataURL, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("MCP-Protocol-Version", "2025-03-26")

	resp, err := am.client.Do(req)
	if err != nil {
		return nil, NewTransportError("http", "metadata discovery", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("metadata discovery failed: HTTP %d", resp.StatusCode)
	}

	var metadata ServerMetadata
	if err := json.NewDecoder(resp.Body).Decode(&metadata); err != nil {
		return nil, fmt.Errorf("failed to decode metadata: %w", err)
	}

	return &metadata, nil
}

// RegisterClient performs dynamic client registration
func (am *AuthManager) RegisterClient(ctx context.Context, registrationEndpoint string, clientName string) (*ClientRegistration, error) {
	registration := map[string]interface{}{
		"client_name":   clientName,
		"redirect_uris": []string{am.config.RedirectURI},
	}

	data, err := json.Marshal(registration)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", registrationEndpoint, strings.NewReader(string(data)))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := am.client.Do(req)
	if err != nil {
		return nil, NewTransportError("http", "client registration", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 201 && resp.StatusCode != 200 {
		return nil, fmt.Errorf("client registration failed: HTTP %d", resp.StatusCode)
	}

	var result ClientRegistration
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode registration response: %w", err)
	}

	// Update config with registered client credentials
	am.config.ClientID = result.ClientID
	am.config.ClientSecret = result.ClientSecret

	return &result, nil
}

// GeneratePKCE generates PKCE code verifier and challenge
func GeneratePKCE() (verifier, challenge string, err error) {
	// Generate 32 random bytes for code verifier
	verifierBytes := make([]byte, 32)
	if _, err := rand.Read(verifierBytes); err != nil {
		return "", "", fmt.Errorf("failed to generate random bytes: %w", err)
	}

	// Base64 URL encode without padding
	verifier = base64.RawURLEncoding.EncodeToString(verifierBytes)

	// Create SHA256 hash of verifier for challenge
	h := sha256.New()
	h.Write([]byte(verifier))
	challengeBytes := h.Sum(nil)

	// Base64 URL encode without padding
	challenge = base64.RawURLEncoding.EncodeToString(challengeBytes)

	return verifier, challenge, nil
}

// GetAuthorizationURL generates the authorization URL
func (am *AuthManager) GetAuthorizationURL(state, codeChallenge string) (string, error) {
	if am.config.AuthorizationEndpoint == "" {
		return "", fmt.Errorf("authorization endpoint not configured")
	}

	if am.config.ClientID == "" {
		return "", fmt.Errorf("client ID not configured")
	}

	params := url.Values{}
	params.Set("response_type", "code")
	params.Set("client_id", am.config.ClientID)
	params.Set("redirect_uri", am.config.RedirectURI)
	params.Set("state", state)
	params.Set("code_challenge", codeChallenge)
	params.Set("code_challenge_method", "S256")

	if len(am.config.Scopes) > 0 {
		params.Set("scope", strings.Join(am.config.Scopes, " "))
	}

	return am.config.AuthorizationEndpoint + "?" + params.Encode(), nil
}

// ExchangeCode exchanges an authorization code for tokens
func (am *AuthManager) ExchangeCode(ctx context.Context, code, codeVerifier string) (*TokenResponse, error) {
	if am.config.TokenEndpoint == "" {
		return nil, fmt.Errorf("token endpoint not configured")
	}

	params := url.Values{}
	params.Set("grant_type", "authorization_code")
	params.Set("code", code)
	params.Set("redirect_uri", am.config.RedirectURI)
	params.Set("code_verifier", codeVerifier)
	params.Set("client_id", am.config.ClientID)

	if am.config.ClientSecret != "" {
		params.Set("client_secret", am.config.ClientSecret)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", am.config.TokenEndpoint, strings.NewReader(params.Encode()))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := am.client.Do(req)
	if err != nil {
		return nil, NewTransportError("http", "token exchange", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, NewAuthenticationError(fmt.Sprintf("token exchange failed: HTTP %d", resp.StatusCode))
	}

	var tokenResp TokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return nil, fmt.Errorf("failed to decode token response: %w", err)
	}

	// Store access token
	am.accessToken = tokenResp.AccessToken
	if tokenResp.ExpiresIn > 0 {
		am.tokenExpiry = time.Now().Add(time.Duration(tokenResp.ExpiresIn) * time.Second)
	}

	return &tokenResp, nil
}

// GetAccessToken returns the current access token
func (am *AuthManager) GetAccessToken() string {
	return am.accessToken
}

// IsTokenExpired checks if the access token is expired
func (am *AuthManager) IsTokenExpired() bool {
	if am.accessToken == "" {
		return true
	}
	if am.tokenExpiry.IsZero() {
		return false // No expiry set
	}
	return time.Now().After(am.tokenExpiry)
}
