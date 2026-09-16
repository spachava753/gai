package gai

import (
	"strings"
	"testing"
)

func requireContentContaining(t *testing.T, resp Response, want string) {
	t.Helper()
	if len(resp.Candidates) == 0 {
		t.Fatal("no candidates returned")
	}
	for _, block := range resp.Candidates[0].Blocks {
		if block.BlockType == Content && strings.Contains(strings.ToLower(block.Content.String()), strings.ToLower(want)) {
			return
		}
	}
	t.Fatalf("no content block contained %q; response: %+v", want, resp)
}

func requireBlockType(t *testing.T, resp Response, blockType string) Block {
	t.Helper()
	if len(resp.Candidates) == 0 {
		t.Fatal("no candidates returned")
	}
	for _, block := range resp.Candidates[0].Blocks {
		if block.BlockType == blockType {
			return block
		}
	}
	t.Fatalf("no %s block found; response: %+v", blockType, resp)
	return Block{}
}
