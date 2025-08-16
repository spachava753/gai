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
	client             *http.Client
	baseURL            string
	apiKey             string
	model              string
	systemInstructions string
	tools              []cerebrasTool
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
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// NewCerebrasGenerator creates a new Cerebras generator.
// If httpClient is nil, http.DefaultClient is used.
// If baseURL is empty, "https://api.cerebras.ai" is used.
// apiKey is read from CEREBRAS_API_KEY if empty.
func NewCerebrasGenerator(httpClient *http.Client, baseURL, model, systemInstructions string, apiKey string) *CerebrasGenerator {
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
		client:             httpClient,
		baseURL:            strings.TrimRight(baseURL, "/"),
		apiKey:             apiKey,
		model:              model,
		systemInstructions: systemInstructions,
		tools:              nil,
	}
}

// Register implements ToolRegister
func (g *CerebrasGenerator) Register(tool Tool) error {
	if tool.Name == "" {
		return &ToolRegistrationErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be empty")}
	}
	if tool.Name == ToolChoiceAuto || tool.Name == ToolChoiceToolsRequired {
		return &ToolRegistrationErr{Tool: tool.Name, Cause: fmt.Errorf("tool name cannot be %s", tool.Name)}
	}
	// Check duplicates
	for _, t := range g.tools {
		if t.Function.Name == tool.Name {
			return &ToolRegistrationErr{Tool: tool.Name, Cause: fmt.Errorf("tool already registered")}
		}
	}

	params := map[string]interface{}{}
	if tool.InputSchema != nil {
		// Marshal to generic map for JSON Schema
		b, err := json.Marshal(tool.InputSchema)
		if err != nil {
			return &ToolRegistrationErr{Tool: tool.Name, Cause: err}
		}
		if err := json.Unmarshal(b, &params); err != nil {
			return &ToolRegistrationErr{Tool: tool.Name, Cause: err}
		}
		// Treat {"type":"object"} as empty parameter list (omit parameters)
		if len(params) == 1 {
			if t, ok := params["type"].(string); ok && t == "object" {
				params = map[string]interface{}{}
			}
		}
	}

	fn := cerebrasFunctionDef{
		Name:        tool.Name,
		Description: tool.Description,
	}
	if len(params) > 0 {
		fn.Parameters = params
	}
	g.tools = append(g.tools, cerebrasTool{
		Type:     "function",
		Function: fn,
	})
	return nil
}

func (g *CerebrasGenerator) buildMessages(dialog Dialog) ([]cerebrasMessage, error) {
	var msgs []cerebrasMessage

	// Add system instructions first if present
	if g.systemInstructions != "" {
		msgs = append(msgs, cerebrasMessage{Role: "system", Content: g.systemInstructions})
	}

	for i, msg := range dialog {
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
			var text strings.Builder
			var toolCalls []cerebrasToolCall
			for _, blk := range msg.Blocks {
				switch blk.BlockType {
				case Content:
					if blk.ModalityType != Text {
						return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
					text.WriteString(blk.Content.String())
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
			cm := cerebrasMessage{Role: "assistant", Content: text.String()}
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
func (g *CerebrasGenerator) Generate(ctx context.Context, dialog Dialog, options *GenOpts) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("cerebras: client not initialized")
	}
	if g.apiKey == "" {
		return Response{}, AuthenticationErr("missing API key")
	}
	if len(dialog) == 0 {
		return Response{}, EmptyDialogErr
	}

	msgs, err := g.buildMessages(dialog)
	if err != nil {
		return Response{}, err
	}

	req := cerebrasChatRequest{
		Model:    g.model,
		Messages: msgs,
	}

	// Map GenOpts subset supported by Cerebras
	if options != nil {
		if options.Temperature != 0 {
			t := options.Temperature
			req.Temperature = &t
		}
		if options.TopP != 0 {
			tp := options.TopP
			req.TopP = &tp
		}
		if options.FrequencyPenalty != 0 {
			return Response{}, fmt.Errorf("frequency penalty is invalid")
		}
		if options.PresencePenalty != 0 {
			return Response{}, fmt.Errorf("presence penalty is invalid")
		}
		if options.TopK != 0 {
			return Response{}, fmt.Errorf("top_k is invalid")
		}
		if options.MaxGenerationTokens > 0 {
			m := options.MaxGenerationTokens
			req.MaxCompletionTokens = &m
		}
		if options.N > 0 {
			return Response{}, fmt.Errorf("n is invalid")
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
		// ThinkingBudget: map low/medium/high to reasoning_effort
		if options.ThinkingBudget != "" {
			switch options.ThinkingBudget {
			case "low", "medium", "high":
				req.ReasoningEffort = options.ThinkingBudget
			default:
				return Response{}, &InvalidParameterErr{Parameter: "thinking budget", Reason: fmt.Sprintf("invalid value: %s", options.ThinkingBudget)}
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

	if len(g.tools) > 0 {
		req.Tools = g.tools
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
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Map common status codes
		switch resp.StatusCode {
		case 401:
			return Response{}, AuthenticationErr(strings.TrimSpace(string(respBody)))
		case 403:
			return Response{}, ApiErr{StatusCode: resp.StatusCode, Type: "permission_error", Message: strings.TrimSpace(string(respBody))}
		case 404:
			return Response{}, ApiErr{StatusCode: resp.StatusCode, Type: "not_found_error", Message: strings.TrimSpace(string(respBody))}
		case 413:
			return Response{}, ApiErr{StatusCode: resp.StatusCode, Type: "request_too_large", Message: strings.TrimSpace(string(respBody))}
		case 429:
			return Response{}, RateLimitErr(strings.TrimSpace(string(respBody)))
		case 500:
			return Response{}, ApiErr{StatusCode: resp.StatusCode, Type: "api_error", Message: strings.TrimSpace(string(respBody))}
		case 503:
			return Response{}, ApiErr{StatusCode: resp.StatusCode, Type: "service_unavailable", Message: strings.TrimSpace(string(respBody))}
		default:
			return Response{}, ApiErr{StatusCode: resp.StatusCode, Type: "invalid_request_error", Message: strings.TrimSpace(string(respBody))}
		}
	}

	var cr cerebrasChatResponse
	if err := json.Unmarshal(respBody, &cr); err != nil {
		return Response{}, fmt.Errorf("failed to parse response: %w", err)
	}

	result := Response{UsageMetrics: make(Metrics)}
	if cr.Usage.PromptTokens > 0 {
		result.UsageMetrics[UsageMetricInputTokens] = cr.Usage.PromptTokens
	}
	if cr.Usage.CompletionTokens > 0 {
		result.UsageMetrics[UsageMetricGenerationTokens] = cr.Usage.CompletionTokens
	}

	var hasToolCalls bool
	for _, ch := range cr.Choices {
		var blocks []Block
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
			return result, MaxGenerationLimitErr
		case "tool_calls":
			result.FinishReason = ToolUse
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
var _ ToolRegister = (*CerebrasGenerator)(nil)
