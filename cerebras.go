package gai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// CerebrasGenerator implements the Generator interface using Cerebras Chat Completions HTTP API
// Endpoint: POST {baseURL}/v1/chat/completions
// No streaming and no token counting support.
type CerebrasGenerator struct {
	client  *http.Client
	baseURL string
	apiKey  string
}

type cerebrasMessage struct {
	Role        string                 `json:"role"`
	Content     string                 `json:"content"`
	Name        string                 `json:"name,omitempty"`
	ToolCallID  string                 `json:"tool_call_id,omitempty"`
	ToolCalls   []cerebrasToolCall     `json:"tool_calls,omitempty"`
	ExtraFields map[string]interface{} `json:"-"`
}

type cerebrasTool struct {
	Type     string              `json:"type"`
	Function cerebrasFunctionDef `json:"function"`
}

type cerebrasFunctionDef struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type cerebrasToolCall struct {
	Type     string                      `json:"type"`
	Function cerebrasToolCallFunctionDef `json:"function"`
	ID       string                      `json:"id,omitempty"`
}

type cerebrasToolCallFunctionDef struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type cerebrasChatRequest struct {
	Model               string                 `json:"model"`
	Messages            []cerebrasMessage      `json:"messages"`
	Temperature         *float64               `json:"temperature,omitempty"`
	TopP                *float64               `json:"top_p,omitempty"`
	MaxCompletionTokens *int                   `json:"max_completion_tokens,omitempty"`
	Stop                any                    `json:"stop,omitempty"`
	Tools               []cerebrasTool         `json:"tools,omitempty"`
	ToolChoice          any                    `json:"tool_choice,omitempty"`
	ReasoningEffort     string                 `json:"reasoning_effort,omitempty"`
	DisableReasoning    *bool                  `json:"disable_reasoning,omitempty"`
	ResponseFormat      map[string]any         `json:"response_format,omitempty"`
	User                string                 `json:"user,omitempty"`
	Seed                *int                   `json:"seed,omitempty"`
	Logprobs            *bool                  `json:"logprobs,omitempty"`
	TopLogprobs         *int                   `json:"top_logprobs,omitempty"`
	Extra               map[string]interface{} `json:"-"`
}

type cerebrasChatResponse struct {
	ID                string `json:"id"`
	Created           int64  `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Object            string `json:"object"`
	Choices           []struct {
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Role      string             `json:"role"`
			Content   string             `json:"content"`
			ToolCalls []cerebrasToolCall `json:"tool_calls,omitempty"`
			Refusal   string             `json:"refusal,omitempty"`
			Reasoning *string            `json:"reasoning,omitempty"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
		// PromptTokensDetails contains detailed token breakdown including cache information.
		// Cerebras supports automatic prompt caching and reports cache read tokens here.
		// Note: Cerebras only reports cache read tokens (cached_tokens), not cache write tokens,
		// since their caching mechanism is automatic and doesn't expose write metrics.
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details,omitempty"`
	} `json:"usage"`
}

// NewCerebrasGenerator creates a stateless Cerebras generator.
// If httpClient is nil, http.DefaultClient is used.
// If baseURL is empty, "https://api.cerebras.ai" is used.
// apiKey is read from CEREBRAS_API_KEY if empty.
func NewCerebrasGenerator(httpClient *http.Client, baseURL, apiKey string) *CerebrasGenerator {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	if baseURL == "" {
		baseURL = "https://api.cerebras.ai"
	}
	if apiKey == "" {
		apiKey = os.Getenv("CEREBRAS_API_KEY")
	}
	return &CerebrasGenerator{
		client:  httpClient,
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
	}
}

func convertToolsToCerebras(tools []Tool) ([]cerebrasTool, error) {
	converted := make([]cerebrasTool, 0, len(tools))
	seen := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		if tool.Name == "" {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be empty")}
		}
		if tool.Name == ToolChoiceAuto || tool.Name == ToolChoiceToolsRequired {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be %s", tool.Name)}
		}
		if _, exists := seen[tool.Name]; exists {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: fmt.Errorf("tool already provided")}
		}
		seen[tool.Name] = struct{}{}

		params := map[string]interface{}{}
		if tool.InputSchema != nil {
			data, err := json.Marshal(tool.InputSchema)
			if err != nil {
				return nil, &InvalidToolErr{Tool: tool.Name, Cause: err}
			}
			if err := json.Unmarshal(data, &params); err != nil {
				return nil, &InvalidToolErr{Tool: tool.Name, Cause: err}
			}
			if len(params) == 1 && params["type"] == "object" {
				params = map[string]interface{}{}
			}
		}

		function := cerebrasFunctionDef{Name: tool.Name, Description: tool.Description}
		if len(params) > 0 {
			function.Parameters = params
		}
		converted = append(converted, cerebrasTool{Type: "function", Function: function})
	}
	return converted, nil
}

type cerebrasGenerationOptions struct {
	Temperature         *float64
	TopP                *float64
	MaxGenerationTokens *int
	ToolChoice          string
	StopSequences       []string
	OutputModalities    []Modality
	ThinkingBudget      string
}

func parseCerebrasGenerationOptions(values GenerationOptions) (*cerebrasGenerationOptions, error) {
	options := &cerebrasGenerationOptions{}

	temperature, ok, err := generationOption[float64](values, GenerationOptionTemperature)
	if err != nil {
		return nil, err
	}
	if ok {
		options.Temperature = &temperature
	}
	topP, ok, err := generationOption[float64](values, GenerationOptionTopP)
	if err != nil {
		return nil, err
	}
	if ok {
		options.TopP = &topP
	}
	maxTokens, ok, err := generationOption[int](values, GenerationOptionMaxGenerationTokens)
	if err != nil {
		return nil, err
	}
	if ok {
		options.MaxGenerationTokens = &maxTokens
	}
	if options.ToolChoice, _, err = generationOption[string](values, GenerationOptionToolChoice); err != nil {
		return nil, err
	}
	if options.StopSequences, _, err = generationOption[[]string](values, GenerationOptionStopSequences); err != nil {
		return nil, err
	}
	if options.OutputModalities, _, err = generationOption[[]Modality](values, GenerationOptionOutputModalities); err != nil {
		return nil, err
	}
	if options.ThinkingBudget, _, err = generationOption[string](values, GenerationOptionThinkingBudget); err != nil {
		return nil, err
	}
	return options, nil
}

func (g *CerebrasGenerator) buildMessages(request GenerationRequest) ([]cerebrasMessage, error) {
	var msgs []cerebrasMessage

	instructions, err := joinedTextInstructions(request.Instructions)
	if err != nil {
		return nil, err
	}
	if instructions != "" {
		msgs = append(msgs, cerebrasMessage{Role: "system", Content: instructions})
	}

	for i, msg := range request.Dialog {
		switch msg.Role {
		case User:
			// Concatenate all text content blocks; error on non-text modalities
			var b strings.Builder
			for _, blk := range msg.Blocks {
				if blk.BlockType != Content {
					return nil, fmt.Errorf("unsupported block type for user: %v", blk.BlockType)
				}
				if blk.ModalityType != Text {
					return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
				}
				b.WriteString(blk.Content.String())
			}
			msgs = append(msgs, cerebrasMessage{Role: "user", Content: b.String()})
		case Assistant:
			var text string
			var toolCalls []cerebrasToolCall
			var reasoningText string
			for _, blk := range msg.Blocks {
				switch blk.BlockType {
				case Content:
					if blk.ModalityType != Text {
						return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
					text = blk.Content.String()
				case Thinking:
					// For Cerebras, reasoning content should be included in the content field
					if blk.ModalityType != Text {
						return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
					reasoningText = blk.Content.String()
				case ToolCall:
					var toolUse ToolCallInput
					if err := json.Unmarshal([]byte(blk.Content.String()), &toolUse); err != nil {
						return nil, fmt.Errorf("invalid tool call content: %w", err)
					}
					argsJSON, err := json.Marshal(toolUse.Parameters)
					if err != nil {
						return nil, fmt.Errorf("failed to marshal tool parameters: %w", err)
					}
					toolCalls = append(toolCalls, cerebrasToolCall{
						Type: "function",
						ID:   blk.ID,
						Function: cerebrasToolCallFunctionDef{
							Name:      toolUse.Name,
							Arguments: string(argsJSON),
						},
					})
				default:
					return nil, fmt.Errorf("unsupported block type for assistant: %v", blk.BlockType)
				}
			}

			// Combine reasoning text with content if present
			if reasoningText != "" {
				text = fmt.Sprintf("<thinking>%s</thinking>\n%s", reasoningText, text)
			}

			cm := cerebrasMessage{Role: "assistant", Content: text}
			if len(toolCalls) > 0 {
				cm.ToolCalls = toolCalls
			}
			msgs = append(msgs, cm)
		case ToolResult:
			if len(msg.Blocks) == 0 {
				return nil, fmt.Errorf("tool result message must have at least one block")
			}
			// Only support text tool results for now
			for _, blk := range msg.Blocks {
				if blk.ModalityType != Text {
					return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
				}
				if blk.ID == "" {
					return nil, fmt.Errorf("tool result message block must have the tool_call_id as ID")
				}
				msgs = append(msgs, cerebrasMessage{
					Role:       "tool",
					Content:    blk.Content.String(),
					ToolCallID: blk.ID,
				})
			}
		default:
			return nil, fmt.Errorf("unsupported role at index %d: %v", i, msg.Role)
		}
	}
	return msgs, nil
}

// Generate implements Generator
func (g *CerebrasGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("cerebras: client not initialized")
	}
	if g.apiKey == "" {
		return Response{}, fmt.Errorf("cerebras: missing API key")
	}
	if len(request.Dialog) == 0 {
		return Response{}, ErrEmptyDialog
	}
	options, err := parseCerebrasGenerationOptions(request.Options)
	if err != nil {
		return Response{}, err
	}
	tools, err := convertToolsToCerebras(request.Tools)
	if err != nil {
		return Response{}, err
	}

	msgs, err := g.buildMessages(request)
	if err != nil {
		return Response{}, err
	}

	req := cerebrasChatRequest{
		Model:    request.Model,
		Messages: msgs,
		Tools:    tools,
	}

	// Map the options supported by Cerebras.
	if options != nil {
		if options.Temperature != nil {
			req.Temperature = options.Temperature
		}
		if options.TopP != nil {
			req.TopP = options.TopP
		}
		if options.MaxGenerationTokens != nil {
			req.MaxCompletionTokens = options.MaxGenerationTokens
		}
		if len(options.StopSequences) > 0 {
			if len(options.StopSequences) == 1 {
				req.Stop = options.StopSequences[0]
			} else {
				req.Stop = options.StopSequences
			}
		}
		if options.ToolChoice != "" {
			switch options.ToolChoice {
			case ToolChoiceAuto:
				req.ToolChoice = "auto"
			case ToolChoiceToolsRequired:
				req.ToolChoice = "required"
			case "none":
				req.ToolChoice = "none"
			default:
				req.ToolChoice = map[string]any{
					"type":     "function",
					"function": map[string]any{"name": options.ToolChoice},
				}
			}
		}
		// ThinkingBudget: handle reasoning parameters based on model
		if options.ThinkingBudget != "" {
			// For gpt-oss-120b model: use reasoning_effort with low/medium/high
			if request.Model == "gpt-oss-120b" {
				switch options.ThinkingBudget {
				case "low", "medium", "high":
					req.ReasoningEffort = options.ThinkingBudget
				default:
					return Response{}, &InvalidParameterErr{Parameter: "thinking budget", Reason: fmt.Sprintf("invalid value for gpt-oss-120b: %s (must be low, medium, or high)", options.ThinkingBudget)}
				}
			} else if request.Model == "zai-glm-4.6" {
				// For zai-glm-4.6 model: if value is false, use disable_reasoning
				if options.ThinkingBudget == "false" {
					disable := false
					req.DisableReasoning = &disable
				}
			}
		}
		// Unsupported output modalities
		if len(options.OutputModalities) > 0 {
			for _, m := range options.OutputModalities {
				if m != Text {
					return Response{}, UnsupportedOutputModalityErr(m.String())
				}
			}
		}
	}

	body, err := json.Marshal(req)
	if err != nil {
		return Response{}, fmt.Errorf("failed to marshal request: %w", err)
	}

	endpoint := g.baseURL + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return Response{}, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+g.apiKey)

	resp, err := g.client.Do(httpReq)
	if err != nil {
		return Response{}, fmt.Errorf("request failed: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Response{}, mapHTTPAPIError(ProviderCerebras, resp)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)

	var cr cerebrasChatResponse
	if err := json.Unmarshal(respBody, &cr); err != nil {
		return Response{}, fmt.Errorf("failed to parse response: %w", err)
	}

	result := Response{UsageMetadata: make(Metadata)}
	if cr.Usage.PromptTokens > 0 {
		result.UsageMetadata[UsageMetricInputTokens] = cr.Usage.PromptTokens
	}
	if cr.Usage.CompletionTokens > 0 {
		result.UsageMetadata[UsageMetricGenerationTokens] = cr.Usage.CompletionTokens
	}
	if cr.Usage.PromptTokensDetails != nil && cr.Usage.PromptTokensDetails.CachedTokens > 0 {
		result.UsageMetadata[UsageMetricCacheReadTokens] = cr.Usage.PromptTokensDetails.CachedTokens
	}

	var hasToolCalls bool
	for _, ch := range cr.Choices {
		if ch.Message.Refusal != "" {
			result.FinishReason = ContentPolicyViolation
			return result, ContentPolicyErr(ch.Message.Refusal)
		}
		var blocks []Block

		// Add reasoning block if present
		if ch.Message.Reasoning != nil && *ch.Message.Reasoning != "" {
			blocks = append(blocks, Block{
				BlockType:    Thinking,
				ModalityType: Text,
				MimeType:     "text/plain",
				Content:      Str(*ch.Message.Reasoning),
				ExtraFields: map[string]interface{}{
					ThinkingExtraFieldGeneratorKey: ThinkingGeneratorCerebras,
				},
			})
		}

		if s := ch.Message.Content; s != "" {
			blocks = append(blocks, Block{BlockType: Content, ModalityType: Text, MimeType: "text/plain", Content: Str(s)})
		}
		if len(ch.Message.ToolCalls) > 0 {
			hasToolCalls = true
			for _, tc := range ch.Message.ToolCalls {
				// Normalize to ToolCallInput JSON
				var params map[string]any
				if tc.Function.Arguments != "" {
					_ = json.Unmarshal([]byte(tc.Function.Arguments), &params)
				}
				tj, _ := json.Marshal(ToolCallInput{Name: tc.Function.Name, Parameters: params})
				blocks = append(blocks, Block{
					ID:           tc.ID,
					BlockType:    ToolCall,
					ModalityType: Text,
					MimeType:     "application/json",
					Content:      Str(tj),
				})
			}
		}
		result.Candidates = append(result.Candidates, Message{Role: Assistant, Blocks: blocks})
	}

	if len(cr.Choices) > 0 {
		switch cr.Choices[0].FinishReason {
		case "stop":
			result.FinishReason = EndTurn
		case "length":
			result.FinishReason = MaxGenerationLimit
			return result, ErrMaxGenerationLimit
		case "tool_calls":
			result.FinishReason = ToolUse
		case "content_filter":
			result.FinishReason = ContentPolicyViolation
			return result, ContentPolicyErr("content policy violation detected")
		default:
			result.FinishReason = Unknown
		}
	}
	if hasToolCalls && result.FinishReason == EndTurn {
		result.FinishReason = ToolUse
	}
	return result, nil
}

var _ Generator = (*CerebrasGenerator)(nil)
