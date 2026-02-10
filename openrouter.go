package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

const (
	// OpenRouterExtraFieldReasoningType stores the reasoning detail type (e.g., "reasoning.summary", "reasoning.text", "reasoning.encrypted").
	// Present in Block.ExtraFields for Thinking blocks from OpenRouter responses.
	OpenRouterExtraFieldReasoningType = "reasoning_type"

	// OpenRouterExtraFieldReasoningFormat stores the reasoning detail format (e.g., "anthropic-claude-v1").
	// Present in Block.ExtraFields for Thinking blocks from OpenRouter responses.
	OpenRouterExtraFieldReasoningFormat = "reasoning_format"

	// OpenRouterExtraFieldReasoningIndex stores the zero-based index of the reasoning detail in the response.
	// Present in Block.ExtraFields for Thinking blocks from OpenRouter responses.
	OpenRouterExtraFieldReasoningIndex = "reasoning_index"

	// OpenRouterExtraFieldReasoningSignature stores the signature for encrypted reasoning details.
	// Present in Block.ExtraFields for Thinking blocks with type "reasoning.text" when a signature is provided.
	OpenRouterExtraFieldReasoningSignature = "reasoning_signature"

	// OpenRouterUsageMetricReasoningDetailsAvailable indicates whether reasoning_details were present in the response.
	// Stored in Response.UsageMetadata as a boolean value.
	OpenRouterUsageMetricReasoningDetailsAvailable = "reasoning_details_available"
)

// OpenRouterGenerator implements the Generator interface using OpenRouter's API,
// which is largely compatible with OpenAI's API but includes additional features
// like reasoning tokens and extended error information.
//
// OpenRouter is a unified API that provides access to multiple LLM providers
// (OpenAI, Anthropic, Google, Meta, etc.) through a single interface. This
// generator leverages the OpenAI SDK since OpenRouter's API is a superset of
// OpenAI's API.
//
// Reasoning Support:
// OpenRouter supports reasoning tokens via the "reasoning" parameter with effort
// levels ("low", "medium", "high") or max_tokens (as string). This generator:
// 1. Sets reasoning config in requests via ThinkingBudget in GenOpts
// 2. Extracts reasoning_details from responses as Thinking blocks with extra fields:
//    - OpenRouterExtraFieldReasoningType
//    - OpenRouterExtraFieldReasoningFormat
//    - OpenRouterExtraFieldReasoningIndex
//    - OpenRouterExtraFieldReasoningSignature (when applicable)
// 3. Passes reasoning_details back in assistant messages (recommended by OpenRouter)
// 4. Sets OpenRouterUsageMetricReasoningDetailsAvailable in Response.UsageMetadata when reasoning_details are present
//
// Note: Streaming is not yet implemented for this generator.
type OpenRouterGenerator struct {
	client             OpenAICompletionService
	model              string
	tools              map[string]oai.ChatCompletionToolUnionParam
	systemInstructions string
}

// openRouterRawResponse wraps the raw JSON response from OpenRouter
type openRouterRawResponse struct {
	Choices []struct {
		Message struct {
			ReasoningDetails []struct {
				Type      string `json:"type"`
				Summary   string `json:"summary,omitempty"`
				Text      string `json:"text,omitempty"`
				Data      string `json:"data,omitempty"` // For encrypted reasoning
				Signature string `json:"signature,omitempty"`
				ID        string `json:"id"`
				Format    string `json:"format"`
				Index     int    `json:"index"`
			} `json:"reasoning_details,omitempty"`
		} `json:"message"`
	} `json:"choices"`
}

// openRouterReasoningDetail represents a single reasoning detail in the request
type openRouterReasoningDetail struct {
	Type      string `json:"type"`
	Summary   string `json:"summary,omitempty"`
	Text      string `json:"text,omitempty"`
	Data      string `json:"data,omitempty"`
	Signature string `json:"signature,omitempty"`
	ID        string `json:"id"`
	Format    string `json:"format"`
	Index     int    `json:"index"`
}

// NewOpenRouterGenerator creates a new OpenRouter generator that uses the OpenAI SDK
// with OpenRouter-specific configuration. The baseURL should be "https://openrouter.ai/api/v1"
// and the apiKey should be your OpenRouter API key.
//
// Example:
//
//	client := openai.NewClient(
//	    option.WithBaseURL("https://openrouter.ai/api/v1"),
//	    option.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
//	)
//	gen := NewOpenRouterGenerator(&client.Chat.Completions, "anthropic/claude-3.5-sonnet", "You are helpful")
func NewOpenRouterGenerator(client OpenAICompletionService, model string, systemInstructions string) *OpenRouterGenerator {
	return &OpenRouterGenerator{
		client:             client,
		model:              model,
		systemInstructions: systemInstructions,
		tools:              make(map[string]oai.ChatCompletionToolUnionParam),
	}
}

// Register implements ToolRegister
func (g *OpenRouterGenerator) Register(tool Tool) error {
	// Validate tool name
	if tool.Name == "" {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool name cannot be empty"),
		}
	}

	// Check for special tool choice values
	if tool.Name == ToolChoiceAuto || tool.Name == ToolChoiceToolsRequired {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool name cannot be %s", tool.Name),
		}
	}

	// Initialize tools map if needed
	if g.tools == nil {
		g.tools = make(map[string]oai.ChatCompletionToolUnionParam)
	}

	// Check for conflicts with existing tools
	if _, exists := g.tools[tool.Name]; exists {
		return &ToolRegistrationErr{
			Tool:  tool.Name,
			Cause: fmt.Errorf("tool already registered"),
		}
	}

	// Convert our tool definition to OpenAI's format and store it
	var err error
	g.tools[tool.Name], err = convertToolToOpenAI(tool)

	return err
}

// buildReasoningDetailsForRequest converts thinking blocks into OpenRouter's reasoning_details format
func buildReasoningDetailsForRequest(thinkingBlocks []Block) []openRouterReasoningDetail {
	var details []openRouterReasoningDetail

	for i, block := range thinkingBlocks {
		if block.BlockType != Thinking {
			continue
		}

		detail := openRouterReasoningDetail{
			ID:     block.ID,
			Index:  i,
			Format: "anthropic-claude-v1", // Default format
		}

		// Extract reasoning type from extra fields
		if reasoningType, ok := block.ExtraFields[OpenRouterExtraFieldReasoningType].(string); ok {
			detail.Type = reasoningType
		} else {
			detail.Type = "reasoning.text" // Default type
		}

		// Extract format from extra fields if available
		if format, ok := block.ExtraFields[OpenRouterExtraFieldReasoningFormat].(string); ok {
			detail.Format = format
		}

		// Extract index from extra fields if available
		if index, ok := block.ExtraFields[OpenRouterExtraFieldReasoningIndex].(int); ok {
			detail.Index = index
		}

		// Set content based on type
		switch detail.Type {
		case "reasoning.summary":
			detail.Summary = block.Content.String()
		case "reasoning.encrypted":
			detail.Data = block.Content.String()
		case "reasoning.text":
			detail.Text = block.Content.String()
			if sig, ok := block.ExtraFields[OpenRouterExtraFieldReasoningSignature].(string); ok {
				detail.Signature = sig
			}
		default:
			detail.Text = block.Content.String()
		}

		details = append(details, detail)
	}

	return details
}

// Generate implements Generator
func (g *OpenRouterGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("openrouter: client not initialized")
	}

	// Check for empty dialog
	if len(dialog) == 0 {
		return Response{}, EmptyDialogErr
	}

	// Convert each message to OpenAI format
	// Filter out Thinking blocks since toOpenAIMessage doesn't know how to handle them
	// We'll add them back via WithJSONSet
	var messages []oai.ChatCompletionMessageParamUnion
	var assistantMsgIndices []int // Track indices of assistant messages with thinking
	var thinkingBlocksByMsg [][]Block

	for i, msg := range dialog {
		// Separate thinking blocks from other blocks
		var thinkingBlocks []Block
		var otherBlocks []Block
		for _, block := range msg.Blocks {
			if block.BlockType == Thinking {
				thinkingBlocks = append(thinkingBlocks, block)
			} else {
				otherBlocks = append(otherBlocks, block)
			}
		}

		// Convert message without thinking blocks
		msgForConversion := msg
		msgForConversion.Blocks = otherBlocks

		oaiMsg, err := toOpenAIMessage(msgForConversion)
		if err != nil {
			return Response{}, fmt.Errorf("failed to convert message: %w", err)
		}
		messages = append(messages, oaiMsg)

		// Track assistant messages with thinking blocks
		if msg.Role == Assistant && len(thinkingBlocks) > 0 {
			assistantMsgIndices = append(assistantMsgIndices, i)
			thinkingBlocksByMsg = append(thinkingBlocksByMsg, thinkingBlocks)
		}
	}

	// Create OpenAI chat completion params
	params := oai.ChatCompletionNewParams{
		Model:    g.model,
		Messages: messages,
	}

	// Add system instructions if present
	if g.systemInstructions != "" {
		params.Messages = append([]oai.ChatCompletionMessageParamUnion{
			oai.SystemMessage(g.systemInstructions),
		}, messages...)
	}

	// Map our options to OpenAI params if options are provided
	if options != nil {
		// Set temperature if non-zero
		if options.Temperature != nil {
			params.Temperature = oai.Float(*options.Temperature)
		}

		// Set top_p if non-zero
		if options.TopP != nil {
			params.TopP = oai.Float(*options.TopP)
		}

		// Set frequency penalty
		if options.FrequencyPenalty != nil {
			params.FrequencyPenalty = oai.Float(*options.FrequencyPenalty)
		}

		// Set presence penalty
		if options.PresencePenalty != nil {
			params.PresencePenalty = oai.Float(*options.PresencePenalty)
		}

		// Set max tokens
		if options.MaxGenerationTokens != nil {
			params.MaxCompletionTokens = oai.Int(int64(*options.MaxGenerationTokens))
		}

		// Set number of responses
		if options.N != nil {
			params.N = oai.Int(int64(*options.N))
		}

		// Set stop sequences if specified
		if len(options.StopSequences) > 0 {
			// OpenAI accepts either a single string or array of strings
			if len(options.StopSequences) == 1 {
				params.Stop = oai.ChatCompletionNewParamsStopUnion{OfString: oai.String(options.StopSequences[0])}
			} else {
				params.Stop = oai.ChatCompletionNewParamsStopUnion{OfStringArray: options.StopSequences}
			}
		}

		// Set tool choice if specified
		if options.ToolChoice != "" {
			switch options.ToolChoice {
			case ToolChoiceAuto:
				params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: oai.String(ToolChoiceAuto)}
			case ToolChoiceToolsRequired:
				params.ToolChoice = oai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: oai.String(ToolChoiceToolsRequired)}
			default:
				// Specific tool name
				params.ToolChoice = oai.ToolChoiceOptionFunctionToolChoice(oai.ChatCompletionNamedToolChoiceFunctionParam{
					Name: options.ToolChoice,
				})
			}
		}
	}

	// Add tools if registered
	if len(g.tools) > 0 {
		for _, tool := range g.tools {
			params.Tools = append(params.Tools, tool)
		}
	}

	// Build request options for OpenRouter-specific features
	var reqOpts []option.RequestOption

	// Add reasoning_details to assistant messages with thinking blocks
	for idx, msgIndex := range assistantMsgIndices {
		thinkingBlocks := thinkingBlocksByMsg[idx]
		reasoningDetails := buildReasoningDetailsForRequest(thinkingBlocks)
		if len(reasoningDetails) > 0 {
			// Use sjson format: messages.{index}.reasoning_details
			jsonPath := fmt.Sprintf("messages.%d.reasoning_details", msgIndex)
			reqOpts = append(reqOpts, option.WithJSONSet(jsonPath, reasoningDetails))
		}
	}

	// Handle ThinkingBudget for reasoning configuration
	if options != nil && options.ThinkingBudget != "" {
		// Try to parse as effort level or max_tokens
		switch options.ThinkingBudget {
		case "low", "medium", "high":
			reqOpts = append(reqOpts, option.WithJSONSet("reasoning.effort", options.ThinkingBudget))
		default:
			// Assume it's a numeric max_tokens value (string will be parsed by API)
			reqOpts = append(reqOpts, option.WithJSONSet("reasoning.max_tokens", options.ThinkingBudget))
		}
	}

	// Make the API call with request options
	resp, err := g.client.New(ctx, params, reqOpts...)
	if err != nil {
		// Handle OpenAI-style errors (non-200 status codes)
		// These occur when: 1) request is invalid or 2) API key/account is out of credits
		var apierr *oai.Error
		if errors.As(err, &apierr) {
			switch apierr.StatusCode {
			case 401:
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "authentication_error",
					Message:    apierr.Error(),
				}
			case 429:
				return Response{}, RateLimitErr(apierr.Error())
			case 500:
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "api_error",
					Message:    apierr.Error(),
				}
			case 503:
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "service_unavailable",
					Message:    apierr.Error(),
				}
			default:
				return Response{}, ApiErr{
					StatusCode: apierr.StatusCode,
					Type:       "invalid_request_error",
					Message:    apierr.Error(),
				}
			}
		}
		return Response{}, fmt.Errorf("failed to create new message: %w", err)
	}

	// Check for OpenRouter upstream errors in response body
	// OpenRouter returns 200 status code but includes error in response body for upstream errors
	if rawJSON := resp.RawJSON(); rawJSON != "" {
		var errorCheck struct {
			Error *struct {
				Code    json.Number `json:"code"`
				Message string      `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(rawJSON), &errorCheck); err == nil && errorCheck.Error != nil {
			// Extract status code from error
			statusCode := 500 // Default to server error
			if code, err := errorCheck.Error.Code.Int64(); err == nil {
				statusCode = int(code)
			}

			// Map to appropriate error type
			switch statusCode {
			case 401:
				return Response{}, ApiErr{
					StatusCode: statusCode,
					Type:       "authentication_error",
					Message:    errorCheck.Error.Message,
				}
			case 429:
				return Response{}, RateLimitErr(errorCheck.Error.Message)
			case 500, 502, 503, 504:
				return Response{}, ApiErr{
					StatusCode: statusCode,
					Type:       "upstream_error",
					Message:    errorCheck.Error.Message,
				}
			default:
				return Response{}, ApiErr{
					StatusCode: statusCode,
					Type:       "upstream_error",
					Message:    errorCheck.Error.Message,
				}
			}
		}
	}

	// Convert OpenAI response to our Response type
	result := Response{
		UsageMetadata: make(Metadata),
	}

	// Add usage metrics if available
	if usage := resp.Usage; usage.PromptTokens > 0 || usage.CompletionTokens > 0 {
		if promptTokens := usage.PromptTokens; promptTokens > 0 {
			result.UsageMetadata[UsageMetricInputTokens] = int(promptTokens)
		}
		if completionTokens := usage.CompletionTokens; completionTokens > 0 {
			result.UsageMetadata[UsageMetricGenerationTokens] = int(completionTokens)
		}
		// Check for cached tokens (OpenRouter uses OpenAI SDK structure)
		if usage.PromptTokensDetails.CachedTokens > 0 {
			result.UsageMetadata[UsageMetricCacheReadTokens] = int(usage.PromptTokensDetails.CachedTokens)
		}
	}

	// Try to extract reasoning details from raw JSON response
	// OpenRouter may include reasoning_details that aren't in the OpenAI SDK structs
	var rawResp openRouterRawResponse
	if rawJSON := resp.RawJSON(); rawJSON != "" {
		if err := json.Unmarshal([]byte(rawJSON), &rawResp); err == nil {
			// Successfully parsed raw response, check for reasoning details
			if len(rawResp.Choices) > 0 && len(rawResp.Choices[0].Message.ReasoningDetails) > 0 {
				result.UsageMetadata[OpenRouterUsageMetricReasoningDetailsAvailable] = true
			}
		}
	}

	var hasToolCalls bool

	// Convert all choices to our Message type
	for i, choice := range resp.Choices {
		// Convert the message content
		var blocks []Block

		// Extract reasoning details from raw response for this choice FIRST
		// Reasoning details should come before content blocks
		if len(rawResp.Choices) > i && len(rawResp.Choices[i].Message.ReasoningDetails) > 0 {
			for _, detail := range rawResp.Choices[i].Message.ReasoningDetails {
				// Add reasoning details as Thinking blocks
				var reasoningContent string
				extraFields := map[string]interface{}{
					OpenRouterExtraFieldReasoningType:   detail.Type,
					OpenRouterExtraFieldReasoningFormat: detail.Format,
					OpenRouterExtraFieldReasoningIndex:  detail.Index,
				}

				switch detail.Type {
				case "reasoning.summary":
					reasoningContent = detail.Summary
				case "reasoning.text":
					reasoningContent = detail.Text
					if detail.Signature != "" {
						extraFields[OpenRouterExtraFieldReasoningSignature] = detail.Signature
					}
				case "reasoning.encrypted":
					reasoningContent = detail.Data
				default:
					continue
				}

				if reasoningContent != "" {
					extraFields[ThinkingExtraFieldGeneratorKey] = ThinkingGeneratorOpenRouter
					blocks = append(blocks, Block{
						ID:           detail.ID,
						BlockType:    Thinking,
						ModalityType: Text,
						MimeType:     "text/plain",
						Content:      Str(reasoningContent),
						ExtraFields:  extraFields,
					})
				}
			}
		}

		// Handle text content AFTER reasoning blocks
		if content := choice.Message.Content; content != "" {
			blocks = append(blocks, Block{
				BlockType:    Content,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str(content),
			})
		}

		// Handle tool calls
		if toolCalls := choice.Message.ToolCalls; len(toolCalls) > 0 {
			hasToolCalls = true
			for _, toolCall := range toolCalls {
				// Create a ToolCallInput with standardized format
				toolUse := ToolCallInput{
					Name: toolCall.Function.Name,
				}

				// Parse the arguments string into a map
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &toolUse.Parameters); err != nil {
					return Response{}, fmt.Errorf("failed to parse tool arguments: %w", err)
				}

				// Marshal back to JSON for consistent representation
				toolUseJSON, err := json.Marshal(toolUse)
				if err != nil {
					return Response{}, fmt.Errorf("failed to marshal tool use: %w", err)
				}

				blocks = append(blocks, Block{
					ID:           toolCall.ID,
					BlockType:    ToolCall,
					ModalityType: Text,
					MimeType:     "application/json",
					Content:      Str(toolUseJSON),
				})
			}
		}

		result.Candidates = append(result.Candidates, Message{
			Role:   Assistant,
			Blocks: blocks,
		})
	}

	// Set finish reason
	if len(resp.Choices) > 0 {
		switch resp.Choices[0].FinishReason {
		case "stop":
			result.FinishReason = EndTurn
		case "length":
			result.FinishReason = MaxGenerationLimit
			return result, MaxGenerationLimitErr
		case "tool_calls":
			result.FinishReason = ToolUse
		case "content_filter":
			result.FinishReason = Unknown
			refusalMessage := resp.Choices[0].Message.Refusal
			if refusalMessage == "" {
				refusalMessage = "content policy violation detected"
			}
			return result, ContentPolicyErr(refusalMessage)
		default:
			result.FinishReason = Unknown
		}
	}

	// Some providers return an EndTurn stop reason, despite being a tool call
	if hasToolCalls && result.FinishReason == EndTurn {
		result.FinishReason = ToolUse
	}

	return result, nil
}
