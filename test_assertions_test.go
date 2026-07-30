package gai

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func requireCandidate(t *testing.T, resp Response) Message {
	t.Helper()
	if len(resp.Candidates) == 0 {
		t.Fatal("expected at least one candidate")
	}
	return resp.Candidates[0]
}

func requireBlock(t *testing.T, msg Message, index int) Block {
	t.Helper()
	if len(msg.Blocks) <= index {
		t.Fatalf("expected block index %d, got %d blocks", index, len(msg.Blocks))
	}
	return msg.Blocks[index]
}

func requireContentBlock(t *testing.T, block Block) string {
	t.Helper()
	if block.BlockType != Content {
		t.Fatalf("block type = %v, want %v", block.BlockType, Content)
	}
	return block.Content.String()
}

func requireTextContains(t *testing.T, got string, wantSubstrings ...string) {
	t.Helper()
	for _, want := range wantSubstrings {
		if !strings.Contains(got, want) {
			t.Fatalf("content %q does not contain %q", got, want)
		}
	}
}

func requireToolCall(t *testing.T, block Block) ToolCallInput {
	t.Helper()
	if block.BlockType != ToolCall {
		t.Fatalf("block type = %v, want %v", block.BlockType, ToolCall)
	}
	var call ToolCallInput
	if err := json.Unmarshal([]byte(block.Content.String()), &call); err != nil {
		t.Fatalf("parse tool call %q: %v", block.Content.String(), err)
	}
	return call
}

func collectToolCalls(t *testing.T, blocks []Block) []ToolCallInput {
	t.Helper()
	var calls []ToolCallInput
	for _, block := range blocks {
		if block.BlockType == ToolCall {
			calls = append(calls, requireToolCall(t, block))
		}
	}
	return calls
}

func requireToolCallWithParam(t *testing.T, calls []ToolCallInput, name, param string, value any) {
	t.Helper()
	for _, call := range calls {
		if call.Name == name && call.Parameters[param] == value {
			return
		}
	}
	t.Fatalf("missing tool call %s with %s=%v in %#v", name, param, value, calls)
}

func requireLiveAPIKey(t testing.TB, env string) string {
	t.Helper()

	if os.Getenv("LIVE_TESTS") == "" {
		t.Skip("LIVE_TESTS not set")
	}

	value := os.Getenv(env)
	if value == "" {
		t.Skipf("%s not set", env)
	}
	return value
}
