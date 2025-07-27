package mcp_test

import (
	"context"
	"fmt"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/spachava753/gai/x/mcp/auth"
	"net/http"
	"os"
	"os/exec"
)

func Example_stdio() {
	wd, err := os.Getwd()
	if err != nil {
		fmt.Println(err)
		return
	}

	cmd := exec.Command(
		"docker",
		"run",
		"-i",
		"--mount",
		fmt.Sprintf("type=bind,src=%s,dst=/projects/workspace", wd),
		"desktopcommander",
		"/projects/workspace",
	)

	ctx := context.Background()

	cmdTransport := mcp.NewCommandTransport(cmd)
	client := mcp.NewClient(&mcp.Implementation{
		Name:    "test-client",
		Version: "test",
	}, nil)

	cs, err := client.Connect(ctx, cmdTransport)
	if err != nil {
		fmt.Println(err)
		return
	}

	tools, err := cs.ListTools(ctx, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	if len(tools.Tools) > 0 {
		fmt.Println("has tools")
		return
	}

	if err = cs.Close(); err != nil {
		fmt.Println(err)
		return
	}
	// Output: has tools
}

func Example_sse() {
	ctx := context.Background()

	sseTransport := mcp.NewSSEClientTransport("https://docs.mcp.cloudflare.com/sse", nil)
	client := mcp.NewClient(&mcp.Implementation{
		Name:    "test-client",
		Version: "test",
	}, nil)

	cs, err := client.Connect(ctx, sseTransport)
	if err != nil {
		fmt.Println(err)
		return
	}

	prompts, err := cs.ListPrompts(ctx, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	if len(prompts.Prompts) > 0 {
		fmt.Println("has prompts")
	}

	tools, err := cs.ListTools(ctx, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	if len(tools.Tools) > 0 {
		fmt.Println("has tools")
	}

	var searchTool *mcp.Tool
	for i := range tools.Tools {
		if tools.Tools[i].Name == "search_cloudflare_documentation" {
			searchTool = tools.Tools[i]
			break
		}
	}

	if searchTool != nil {
		args := map[string]interface{}{"query": "what is a worker?"}
		result, err := cs.CallTool(ctx, &mcp.CallToolParams{
			Name:      searchTool.Name,
			Arguments: args,
		})

		if err != nil {
			fmt.Println("Tool call failed:", err)
		}

		if len(result.Content) > 0 {
			fmt.Println("called tool")
		}
	}

	if err = cs.Close(); err != nil {
		fmt.Println(err)
		return
	}
	// Output: has prompts
	// has tools
	// called tool
}

func Example_streamable() {
	ctx := context.Background()

	streamableTransport := mcp.NewStreamableClientTransport("https://mcp.deepwiki.com/mcp", nil)
	client := mcp.NewClient(&mcp.Implementation{
		Name:    "test-client",
		Version: "test",
	}, nil)

	cs, err := client.Connect(ctx, streamableTransport)
	if err != nil {
		fmt.Println(err)
		return
	}

	tools, err := cs.ListTools(ctx, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	if len(tools.Tools) > 0 {
		fmt.Println("has tools")
	}

	var searchTool *mcp.Tool
	for i := range tools.Tools {
		if tools.Tools[i].Name == "read_wiki_structure" {
			searchTool = tools.Tools[i]
			break
		}
	}

	if searchTool != nil {
		args := map[string]interface{}{"repoName": "spachava753/gai"}
		result, err := cs.CallTool(ctx, &mcp.CallToolParams{
			Name:      searchTool.Name,
			Arguments: args,
		})

		if err != nil {
			fmt.Println("Tool call failed:", err)
		}

		if len(result.Content) > 0 {
			fmt.Println("called tool")
		}
	}

	if err = cs.Close(); err != nil {
		fmt.Println(err)
		return
	}
	// Output: has tools
	// called tool
}

func Example_sseAuth() {
	ctx := context.Background()

	serverUrl := "https://ai-gateway.mcp.cloudflare.com/sse"

	redirectPort := "18456"

	sseTransport := mcp.NewSSEClientTransport(serverUrl, &mcp.SSEClientTransportOptions{
		HTTPClient: &http.Client{
			Transport: &auth.Transport{
				Base:         http.DefaultTransport,
				ClientName:   "gai-test-client",
				RedirectUri:  "http://localhost:" + redirectPort + "/callback",
				RedirectPort: redirectPort,
			},
		},
	})
	client := mcp.NewClient(&mcp.Implementation{
		Name:    "test-client",
		Version: "test",
	}, nil)

	cs, err := client.Connect(ctx, sseTransport)
	if err != nil {
		fmt.Println(err)
		return
	}

	tools, err := cs.ListTools(ctx, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	if len(tools.Tools) > 0 {
		fmt.Println("has tools")
	}

	var listGateways *mcp.Tool
	for i := range tools.Tools {
		if tools.Tools[i].Name == "list_gateways" {
			listGateways = tools.Tools[i]
			break
		}
	}

	if listGateways != nil {
		result, err := cs.CallTool(ctx, &mcp.CallToolParams{
			Name: listGateways.Name,
		})

		if err != nil {
			fmt.Println("Tool call failed:", err)
		}

		if len(result.Content) > 0 {
			fmt.Println("called tool")
		}
	}

	if err = cs.Close(); err != nil {
		fmt.Println(err)
		return
	}
	// Output: has tools
	// called tool
}
