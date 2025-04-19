package gai

import (
	"context"
)

// preprocessToolResults consolidates consecutive tool result messages that respond to tool calls
// from the same previous assistant message, adapting OpenAI-style parallel tool use to providers that expect consolidation (e.g. Anthropic, Gemini).
// This generalizes the logic for any tool-capable generator.
func preprocessToolResults(dialog Dialog) Dialog {
	if len(dialog) <= 1 {
		return dialog
	}

	result := make(Dialog, 0, len(dialog))
	i := 0
	for i < len(dialog) {
		if dialog[i].Role != ToolResult {
			result = append(result, dialog[i])
			i++
			continue
		}

		// Find previous assistant w/ tool calls
		assistantMsgIndex := -1
		for j := i - 1; j >= 0; j-- {
			if dialog[j].Role == Assistant {
				hasToolCalls := false
				for _, block := range dialog[j].Blocks {
					if block.BlockType == ToolCall {
						hasToolCalls = true
						break
					}
				}
				if hasToolCalls {
					assistantMsgIndex = j
					break
				}
			}
		}
		if assistantMsgIndex == -1 {
			result = append(result, dialog[i])
			i++
			continue
		}
		var toolCallIDs []string
		for _, block := range dialog[assistantMsgIndex].Blocks {
			if block.BlockType == ToolCall {
				toolCallIDs = append(toolCallIDs, block.ID)
			}
		}
		if len(toolCallIDs) <= 1 {
			result = append(result, dialog[i])
			i++
			continue
		}
		startIndex := i
		j := i + 1
		for j < len(dialog) && dialog[j].Role == ToolResult {
			j++
		}
		if j-startIndex > 1 {
			toolResultMessagesByToolID := make(map[string][]Block)
			for k := startIndex; k < j; k++ {
				for _, block := range dialog[k].Blocks {
					if block.ID != "" {
						toolResultMessagesByToolID[block.ID] = append(toolResultMessagesByToolID[block.ID], block)
					}
				}
			}
			isThisParallelToolUse := false
			for _, id := range toolCallIDs {
				if _, found := toolResultMessagesByToolID[id]; found {
					isThisParallelToolUse = true
					break
				}
			}
			if isThisParallelToolUse {
				var consolidatedBlocks []Block
				anyError := false
				for k := startIndex; k < j; k++ {
					consolidatedBlocks = append(consolidatedBlocks, dialog[k].Blocks...)
					if dialog[k].ToolResultError {
						anyError = true
					}
				}
				result = append(result, Message{
					Role:            ToolResult,
					Blocks:          consolidatedBlocks,
					ToolResultError: anyError,
				})
				i = j
				continue
			}
		}
		result = append(result, dialog[i])
		i++
	}
	return result
}

// PreprocessingGenerator is a transparent wrapper for any ToolCapableGenerator that
// automatically preprocesses the dialog before every Generate call.
//
// Specifically, it consolidates parallel tool result messages into the format required by
// LLM providers such as Anthropic and Gemini, which expect parallel tool results to be
// delivered in a single message with multiple blocks, whereas OpenAI-style dialogs use
// separate messages for each. This wrapper ensures the dialog structure fed into the
// underlying generator is always in the correct, provider-specific format.
//
// This helps keep generator implementations clean, centralizes parallel tool result
// normalization, and can be easily composed with future generators needing the same behavior.
type PreprocessingGenerator struct {
	Inner ToolCapableGenerator
}

var _ ToolCapableGenerator = (*PreprocessingGenerator)(nil)

func (p *PreprocessingGenerator) Register(tool Tool) error {
	return p.Inner.Register(tool)
}

func (p *PreprocessingGenerator) Generate(ctx context.Context, dialog Dialog, opts *GenOpts) (Response, error) {
	processed := preprocessToolResults(dialog)
	return p.Inner.Generate(ctx, processed, opts)
}
