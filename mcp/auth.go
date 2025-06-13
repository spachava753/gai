package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

// ServerMetadata represents OAuth authorization server metadata
type ServerMetadata struct {
	Issuer                            string   `json:"issuer"`
	AuthorizationEndpoint             string   `json:"authorization_endpoint"`
	TokenEndpoint                     string   `json:"token_endpoint"`
	RegistrationEndpoint              string   `json:"registration_endpoint"`
	ResponseTypesSupported            []string `json:"response_types_supported"`
	ResponseModesSupported            []string `json:"response_modes_supported"`
	GrantTypesSupported               []string `json:"grant_types_supported"`
	TokenEndpointAuthMethodsSupported []string `json:"token_endpoint_auth_methods_supported"`
	RevocationEndpoint                string   `json:"revocation_endpoint"`
	CodeChallengeMethodsSupported     []string `json:"code_challenge_methods_supported"`
}

type ResourceMetadata struct {
	ResourceName           string   `json:"resource_name"`
	Resource               string   `json:"resource"`
	AuthorizationServers   []string `json:"authorization_servers"`
	BearerMethodsSupported []string `json:"bearer_methods_supported"`
	ScopesSupported        []string `json:"scopes_supported"`
}

// DynamicClient represents a dynamically registered OAuth client
type DynamicClient struct {
	ClientID     string   `json:"client_id"`
	ClientSecret string   `json:"client_secret,omitempty"`
	RedirectURIs []string `json:"redirect_uris"`
	ClientName   string   `json:"client_name"`
}

func DefaultMetadataEndpoint(serverUrl string) (string, error) {
	const endpoint = "/.well-known/oauth-authorization-server"
	// Parse server URL to get base URL
	u, err := url.Parse(serverUrl)
	if err != nil {
		return "", fmt.Errorf("invalid server URL: %w", err)
	}

	// Authorization base URL is the scheme + host (no path)
	baseURL := fmt.Sprintf("%s://%s", u.Scheme, u.Host)
	metadataURL := baseURL + endpoint
	return metadataURL, err
}

// DiscoverServerMetadata discovers OAuth server metadata
func DiscoverServerMetadata(
	ctx context.Context,
	discoveryUrl string,
	protocolVersion string,
	httpClient interface {
		Do(req *http.Request) (*http.Response, error)
	},
) (*ServerMetadata, error) {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, discoveryUrl, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("MCP-Protocol-Version", protocolVersion)

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, NewTransportError("http", "metadata discovery", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("metadata discovery failed: HTTP %d", resp.StatusCode)
	}

	var metadata ServerMetadata
	if err = json.NewDecoder(resp.Body).Decode(&metadata); err != nil {
		return nil, fmt.Errorf("failed to decode metadata: %w", err)
	}

	return &metadata, nil
}

func FallbackServerMetadata(serverUrl string) (*ServerMetadata, error) {
	// Parse server URL to get base URL
	u, err := url.Parse(serverUrl)
	if err != nil {
		return nil, fmt.Errorf("invalid server URL: %w", err)
	}

	// Authorization base URL is the scheme + host (no path)
	baseURL := fmt.Sprintf("%s://%s", u.Scheme, u.Host)
	var metadata ServerMetadata
	metadata.AuthorizationEndpoint = baseURL + "/authorize"
	metadata.TokenEndpoint = baseURL + "/token"
	metadata.RegistrationEndpoint = baseURL + "/register"
	return &metadata, nil
}

// DynamicRegistration performs dynamic client registration
func DynamicRegistration(
	ctx context.Context,
	registrationEndpoint string,
	redirectUris []string,
	clientName string,
	httpClient interface {
		Do(req *http.Request) (*http.Response, error)
	},
) (*DynamicClient, error) {
	registration := map[string]interface{}{
		"client_name":   clientName,
		"redirect_uris": redirectUris,
	}

	data, err := json.Marshal(registration)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, registrationEndpoint, strings.NewReader(string(data)))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, NewTransportError("http", "client registration", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 201 && resp.StatusCode != 200 {
		return nil, fmt.Errorf("client registration failed: HTTP %d", resp.StatusCode)
	}

	var result DynamicClient
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode registration response: %w", err)
	}

	return &result, nil
}
