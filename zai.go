package gai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"math"
	"net/http"
	"strings"

	"github.com/go-faster/jx"

	"github.com/spachava753/gai/internal/zai"
)

// ZaiGenerator adapts Z.AI Chat Completions to [Generator],
// [StreamingGenerator], and [TokenCounter]. It accepts text plus supported image, video, and PDF URL
// input, produces text, supports function tools, and preserves interleaved
// reasoning for assistant replay.
//
// Z.AI consumes [WithTemperature], [WithTopP],
// [WithMaxGenerationTokens], [WithToolChoice], [WithStopSequences],
// [WithOutputModalities], and [WithThinkingBudget], plus [WithZaiThinking] and
// [WithZaiClearThinking]. Remote media uses [ZaiExtraFieldURL]. Invocation data
// and web-search results use the ZaiResponseExtraField constants in
// [Response.ExtraFields].
type ZaiGenerator struct {
	client *zai.Client
}

const (
	// ZaiGenerationOptionThinkingEnabled is the bool [GenerationOptions] key set
	// by [WithZaiThinking].
	ZaiGenerationOptionThinkingEnabled = "zai_thinking_enabled"
	// ZaiGenerationOptionClearThinking is the bool [GenerationOptions] key set by
	// [WithZaiClearThinking].
	ZaiGenerationOptionClearThinking = "zai_clear_thinking"
)

// WithZaiThinking returns a [GenerationOption] that enables or disables Z.AI
// thinking through [ZaiGenerationOptionThinkingEnabled].
func WithZaiThinking(enabled bool) GenerationOption {
	return func(options GenerationOptions) {
		options[ZaiGenerationOptionThinkingEnabled] = enabled
	}
}

// WithZaiClearThinking returns a [GenerationOption] that controls whether Z.AI
// clears reasoning from earlier turns through
// [ZaiGenerationOptionClearThinking].
func WithZaiClearThinking(clear bool) GenerationOption {
	return func(options GenerationOptions) {
		options[ZaiGenerationOptionClearThinking] = clear
	}
}

const (
	// ZaiDefaultBaseURL is the endpoint used by [NewZaiGenerator] when baseURL is
	// empty.
	ZaiDefaultBaseURL = string(zai.DefaultServer)

	// ZaiExtraFieldURL is the string [Block.ExtraFields] key for remote image,
	// video, or PDF input. Z.AI requires PDF inputs to use a URL.
	ZaiExtraFieldURL = "zai_url"
)

const (
	// ZaiResponseExtraFieldID is the string completion-ID key in
	// [Response.ExtraFields].
	ZaiResponseExtraFieldID = "zai_id"
	// ZaiResponseExtraFieldRequestID is the string request-ID key in
	// [Response.ExtraFields].
	ZaiResponseExtraFieldRequestID = "zai_request_id"
	// ZaiResponseExtraFieldCreated is the int64 Unix timestamp key in
	// [Response.ExtraFields].
	ZaiResponseExtraFieldCreated = "zai_created"
	// ZaiResponseExtraFieldModel is the string response-model key in
	// [Response.ExtraFields].
	ZaiResponseExtraFieldModel = "zai_model"
	// ZaiResponseExtraFieldWebSearchResults is the []map[string]any hosted-search
	// result key in [Response.ExtraFields].
	ZaiResponseExtraFieldWebSearchResults = "zai_web_search_results"
)

// NewZaiGenerator constructs a stateless Z.AI adapter. A nil httpClient uses
// the generated client's default transport, an empty baseURL uses
// [ZaiDefaultBaseURL], and an empty apiKey returns [ErrMissingAPIKey].
func NewZaiGenerator(httpClient *http.Client, baseURL, apiKey string) (*ZaiGenerator, error) {
	if baseURL == "" {
		baseURL = ZaiDefaultBaseURL
	}
	if apiKey == "" {
		return nil, fmt.Errorf("zai: %w", ErrMissingAPIKey)
	}
	options := make([]zai.ClientOption, 0, 1)
	if httpClient != nil {
		options = append(options, zai.WithClient(httpClient))
	}
	client, err := zai.NewClient(baseURL, zaiSecuritySource{apiKey: apiKey}, options...)
	if err != nil {
		return nil, fmt.Errorf("zai: create client: %w", err)
	}
	return &ZaiGenerator{client: client}, nil
}

type zaiSecuritySource struct {
	apiKey string
}

func (s zaiSecuritySource) BearerAuth(ctx context.Context, operationName zai.OperationName) (zai.BearerAuth, error) {
	return zai.BearerAuth{Token: s.apiKey}, nil
}

func convertToolToZai(tool Tool) (zai.FunctionToolSchema, error) {
	parameters := zai.FunctionParameters{}
	if tool.InputSchema != nil {
		schemaJSON, err := json.Marshal(tool.InputSchema)
		if err != nil {
			return zai.FunctionToolSchema{}, err
		}
		var rawParams map[string]json.RawMessage
		if err := json.Unmarshal(schemaJSON, &rawParams); err != nil {
			return zai.FunctionToolSchema{}, err
		}
		if typ, ok := rawParams["type"]; len(rawParams) == 1 && ok && string(typ) == `"object"` {
			rawParams = nil
		}
		for name, raw := range rawParams {
			parameters[name] = jx.Raw(raw)
		}
	}

	return zai.FunctionToolSchema{
		Type: zai.FunctionToolSchemaTypeFunction,
		Function: zai.FunctionObject{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  parameters,
		},
	}, nil
}

func convertToolsToZai(tools []Tool) ([]zai.FunctionToolSchema, error) {
	converted := make([]zai.FunctionToolSchema, 0, len(tools))
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

		providerTool, err := convertToolToZai(tool)
		if err != nil {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: err}
		}
		converted = append(converted, providerTool)
	}
	return converted, nil
}

type zaiGenerationOptions struct {
	Temperature         *float64
	TopP                *float64
	MaxGenerationTokens *int
	ToolChoice          string
	StopSequences       []string
	OutputModalities    []Modality
	ThinkingBudget      string
	ThinkingEnabled     bool
	ClearThinking       bool
}

// parseZaiGenerationOptions validates common and Z.AI-specific values into typed request settings.
func parseZaiGenerationOptions(values GenerationOptions) (*zaiGenerationOptions, error) {
	options := &zaiGenerationOptions{ThinkingEnabled: true, ClearThinking: true}

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
	if options.ThinkingEnabled, _, err = generationOption[bool](values, ZaiGenerationOptionThinkingEnabled); err != nil {
		return nil, err
	} else if _, exists := values[ZaiGenerationOptionThinkingEnabled]; !exists {
		options.ThinkingEnabled = true
	}
	if options.ClearThinking, _, err = generationOption[bool](values, ZaiGenerationOptionClearThinking); err != nil {
		return nil, err
	} else if _, exists := values[ZaiGenerationOptionClearThinking]; !exists {
		options.ClearThinking = true
	}
	return options, nil
}

// buildTextMessages translates instructions and dialog blocks into Z.AI text-completion messages.
func (g *ZaiGenerator) buildTextMessages(dialog Dialog, instructions string) ([]zai.ChatCompletionTextRequestMessagesItem, error) {
	var messages []zai.ChatCompletionTextRequestMessagesItem

	if instructions != "" {
		messages = append(messages, zai.NewChatCompletionTextRequestMessagesItem1ChatCompletionTextRequestMessagesItem(zai.ChatCompletionTextRequestMessagesItem1{
			Role:    zai.ChatCompletionTextRequestMessagesItem1RoleSystem,
			Content: instructions,
		}))
	}

	for i, msg := range dialog {
		switch msg.Role {
		case User:
			var textContent strings.Builder
			for _, blk := range msg.Blocks {
				if blk.BlockType != Content {
					return nil, fmt.Errorf("unsupported block type for user: %v", blk.BlockType)
				}
				if blk.ModalityType != Text {
					return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
				}
				textContent.WriteString(blk.Content.String())
			}
			messages = append(messages, zai.NewChatCompletionTextRequestMessagesItem0ChatCompletionTextRequestMessagesItem(zai.ChatCompletionTextRequestMessagesItem0{
				Role:    zai.ChatCompletionTextRequestMessagesItem0RoleUser,
				Content: textContent.String(),
			}))

		case Assistant:
			var content string
			var toolCalls []zai.ChatCompletionTextRequestMessagesItem2ToolCallsItem

			for _, blk := range msg.Blocks {
				switch blk.BlockType {
				case Content:
					if blk.ModalityType != Text {
						return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
					content = blk.Content.String()
				case Thinking:
					if blk.ModalityType != Text {
						return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
					}
					// The generated text-message schema does not expose reasoning_content on request messages.
				case ToolCall:
					var toolUse ToolCallInput
					if err := json.Unmarshal([]byte(blk.Content.String()), &toolUse); err != nil {
						return nil, fmt.Errorf("invalid tool call content: %w", err)
					}
					argsJSON, err := json.Marshal(toolUse.Parameters)
					if err != nil {
						return nil, fmt.Errorf("failed to marshal tool parameters: %w", err)
					}
					toolCalls = append(toolCalls, zai.ChatCompletionTextRequestMessagesItem2ToolCallsItem{
						ID:   blk.ID,
						Type: zai.ChatCompletionTextRequestMessagesItem2ToolCallsItemTypeFunction,
						Function: zai.NewOptChatCompletionTextRequestMessagesItem2ToolCallsItemFunction(zai.ChatCompletionTextRequestMessagesItem2ToolCallsItemFunction{
							Name:      toolUse.Name,
							Arguments: string(argsJSON),
						}),
					})
				default:
					return nil, fmt.Errorf("unsupported block type for assistant: %v", blk.BlockType)
				}
			}

			assistantMsg := zai.ChatCompletionTextRequestMessagesItem2{
				Role:      zai.ChatCompletionTextRequestMessagesItem2RoleAssistant,
				ToolCalls: toolCalls,
			}
			if content != "" {
				assistantMsg.Content = zai.NewOptString(content)
			}
			messages = append(messages, zai.NewChatCompletionTextRequestMessagesItem2ChatCompletionTextRequestMessagesItem(assistantMsg))

		case ToolResult:
			if len(msg.Blocks) == 0 {
				return nil, fmt.Errorf("tool result message must have at least one block")
			}
			for _, blk := range msg.Blocks {
				if blk.ModalityType != Text {
					return nil, UnsupportedInputModalityErr(blk.ModalityType.String())
				}
				if blk.ID == "" {
					return nil, fmt.Errorf("tool result message block must have the tool_call_id as ID")
				}
				messages = append(messages, zai.NewChatCompletionTextRequestMessagesItem3ChatCompletionTextRequestMessagesItem(zai.ChatCompletionTextRequestMessagesItem3{
					Role:       zai.ChatCompletionTextRequestMessagesItem3RoleTool,
					Content:    blk.Content.String(),
					ToolCallID: blk.ID,
				}))
			}

		default:
			return nil, fmt.Errorf("unsupported role at index %d: %v", i, msg.Role)
		}
	}
	return messages, nil
}

func (g *ZaiGenerator) dialogNeedsVision(dialog Dialog) bool {
	for _, msg := range dialog {
		for _, block := range msg.Blocks {
			if block.BlockType == Content && block.ModalityType != Text {
				return true
			}
		}
	}
	return false
}

func (g *ZaiGenerator) buildVisionMessages(dialog Dialog, instructions string) ([]zai.ChatCompletionVisionRequestMessagesItem, error) {
	var messages []zai.ChatCompletionVisionRequestMessagesItem
	if instructions != "" {
		messages = append(messages, zai.NewChatCompletionVisionRequestMessagesItem1ChatCompletionVisionRequestMessagesItem(zai.ChatCompletionVisionRequestMessagesItem1{
			Role:    zai.ChatCompletionVisionRequestMessagesItem1RoleSystem,
			Content: instructions,
		}))
	}

	for i, msg := range dialog {
		switch msg.Role {
		case User:
			content, err := zaiVisionUserContent(msg.Blocks)
			if err != nil {
				return nil, err
			}
			messages = append(messages, zai.NewChatCompletionVisionRequestMessagesItem0ChatCompletionVisionRequestMessagesItem(zai.ChatCompletionVisionRequestMessagesItem0{
				Role:    zai.ChatCompletionVisionRequestMessagesItem0RoleUser,
				Content: content,
			}))
		case Assistant:
			content, err := zaiAssistantTextContent(msg.Blocks)
			if err != nil {
				return nil, err
			}
			assistantMsg := zai.ChatCompletionVisionRequestMessagesItem2{
				Role: zai.ChatCompletionVisionRequestMessagesItem2RoleAssistant,
			}
			if content != "" {
				assistantMsg.Content = zai.NewOptString(content)
			}
			messages = append(messages, zai.NewChatCompletionVisionRequestMessagesItem2ChatCompletionVisionRequestMessagesItem(assistantMsg))
		case ToolResult:
			return nil, fmt.Errorf("zai: vision requests do not support tool result messages")
		default:
			return nil, fmt.Errorf("unsupported role at index %d: %v", i, msg.Role)
		}
	}
	return messages, nil
}

func zaiAssistantTextContent(blocks []Block) (string, error) {
	var text strings.Builder
	for _, block := range blocks {
		switch block.BlockType {
		case Content:
			if block.ModalityType != Text {
				return "", UnsupportedInputModalityErr(block.ModalityType.String())
			}
			text.WriteString(block.Content.String())
		case Thinking:
			if block.ModalityType != Text {
				return "", UnsupportedInputModalityErr(block.ModalityType.String())
			}
		case ToolCall:
			return "", fmt.Errorf("zai: vision requests do not support assistant tool call messages")
		default:
			return "", fmt.Errorf("unsupported block type for assistant: %v", block.BlockType)
		}
	}
	return text.String(), nil
}

func zaiVisionUserContent(blocks []Block) (zai.ChatCompletionVisionRequestMessagesItem0Content, error) {
	allText := true
	for _, block := range blocks {
		if block.BlockType != Content {
			return zai.ChatCompletionVisionRequestMessagesItem0Content{}, fmt.Errorf("unsupported block type for user: %v", block.BlockType)
		}
		if block.ModalityType != Text {
			allText = false
		}
	}
	if allText {
		var text strings.Builder
		for _, block := range blocks {
			text.WriteString(block.Content.String())
		}
		return zai.NewStringChatCompletionVisionRequestMessagesItem0Content(text.String()), nil
	}

	parts := make([]zai.VisionMultimodalContentItem, 0, len(blocks))
	for _, block := range blocks {
		part, err := zaiVisionContentPart(block)
		if err != nil {
			return zai.ChatCompletionVisionRequestMessagesItem0Content{}, err
		}
		parts = append(parts, part)
	}
	return zai.NewVisionMultimodalContentItemArrayChatCompletionVisionRequestMessagesItem0Content(parts), nil
}

func zaiVisionContentPart(block Block) (zai.VisionMultimodalContentItem, error) {
	switch block.ModalityType {
	case Text:
		return zai.NewVisionMultimodalContentItem0VisionMultimodalContentItem(zai.VisionMultimodalContentItem0{
			Type: zai.VisionMultimodalContentItem0TypeText,
			Text: block.Content.String(),
		}), nil
	case Image:
		if block.MimeType == "application/pdf" {
			url, err := zaiRemoteURL(block)
			if err != nil {
				return zai.VisionMultimodalContentItem{}, fmt.Errorf("zai: PDF inputs require a remote URL in Content or %s: %w", ZaiExtraFieldURL, err)
			}
			return zai.NewVisionMultimodalContentItem3VisionMultimodalContentItem(zai.VisionMultimodalContentItem3{
				Type: zai.VisionMultimodalContentItem3TypeFileURL,
				FileURL: zai.VisionMultimodalContentItem3FileURL{
					URL: url,
				},
			}), nil
		}
		url := zaiImageURL(block)
		return zai.NewVisionMultimodalContentItem1VisionMultimodalContentItem(zai.VisionMultimodalContentItem1{
			Type: zai.VisionMultimodalContentItem1TypeImageURL,
			ImageURL: zai.VisionMultimodalContentItem1ImageURL{
				URL: url,
			},
		}), nil
	case Video:
		url, err := zaiRemoteURL(block)
		if err != nil {
			return zai.VisionMultimodalContentItem{}, fmt.Errorf("zai: video inputs require a remote URL in Content or %s: %w", ZaiExtraFieldURL, err)
		}
		return zai.NewVisionMultimodalContentItem2VisionMultimodalContentItem(zai.VisionMultimodalContentItem2{
			Type: zai.VisionMultimodalContentItem2TypeVideoURL,
			VideoURL: zai.VisionMultimodalContentItem2VideoURL{
				URL: url,
			},
		}), nil
	case Audio:
		return zai.VisionMultimodalContentItem{}, UnsupportedInputModalityErr(block.ModalityType.String())
	default:
		return zai.VisionMultimodalContentItem{}, UnsupportedInputModalityErr(block.ModalityType.String())
	}
}

func zaiImageURL(block Block) string {
	if url := zaiURLField(block); url != "" {
		return url
	}
	content := strings.TrimSpace(block.Content.String())
	if strings.HasPrefix(content, "http://") || strings.HasPrefix(content, "https://") || strings.HasPrefix(content, "data:") {
		return content
	}
	return "data:" + block.MimeType + ";base64," + content
}

func zaiRemoteURL(block Block) (string, error) {
	if url := zaiURLField(block); url != "" {
		if isZaiRemoteURL(url) {
			return url, nil
		}
		return "", fmt.Errorf("not a remote URL: %q", url)
	}
	content := strings.TrimSpace(block.Content.String())
	if isZaiRemoteURL(content) {
		return content, nil
	}
	return "", fmt.Errorf("not a remote URL")
}

func zaiURLField(block Block) string {
	if block.ExtraFields == nil {
		return ""
	}
	for _, key := range []string{ZaiExtraFieldURL, "url"} {
		if v, ok := block.ExtraFields[key].(string); ok && strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func isZaiRemoteURL(s string) bool {
	return strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://")
}

func (g *ZaiGenerator) buildRequest(generationRequest GenerationRequest, stream bool) (zai.PaasV4ChatCompletionsPostReq, zai.PaasV4ChatCompletionsPostParams, error) {
	options, err := parseZaiGenerationOptions(generationRequest.Options)
	if err != nil {
		return zai.PaasV4ChatCompletionsPostReq{}, zai.PaasV4ChatCompletionsPostParams{}, err
	}
	for _, modality := range options.OutputModalities {
		if modality != Text {
			return zai.PaasV4ChatCompletionsPostReq{}, zai.PaasV4ChatCompletionsPostParams{}, UnsupportedOutputModalityErr(modality.String())
		}
	}
	tools, err := convertToolsToZai(generationRequest.Tools)
	if err != nil {
		return zai.PaasV4ChatCompletionsPostReq{}, zai.PaasV4ChatCompletionsPostParams{}, err
	}
	if err := validateZaiToolChoice(options.ToolChoice); err != nil {
		return zai.PaasV4ChatCompletionsPostReq{}, zai.PaasV4ChatCompletionsPostParams{}, err
	}
	instructions, err := joinedTextInstructions(generationRequest.Instructions)
	if err != nil {
		return zai.PaasV4ChatCompletionsPostReq{}, zai.PaasV4ChatCompletionsPostParams{}, err
	}

	params := zai.PaasV4ChatCompletionsPostParams{
		AcceptLanguage: zai.NewOptAcceptLanguage(zai.AcceptLanguageEnUSEn),
	}
	if g.dialogNeedsVision(generationRequest.Dialog) {
		request, err := g.buildVisionRequest(generationRequest.Model, generationRequest.Dialog, instructions, tools, options, stream)
		if err != nil {
			return zai.PaasV4ChatCompletionsPostReq{}, zai.PaasV4ChatCompletionsPostParams{}, err
		}
		return zai.NewChatCompletionVisionRequestPaasV4ChatCompletionsPostReq(request), params, nil
	}

	request, err := g.buildTextRequest(generationRequest.Model, generationRequest.Dialog, instructions, tools, options, stream)
	if err != nil {
		return zai.PaasV4ChatCompletionsPostReq{}, zai.PaasV4ChatCompletionsPostParams{}, err
	}
	return zai.NewChatCompletionTextRequestPaasV4ChatCompletionsPostReq(request), params, nil
}

func validateZaiToolChoice(choice string) error {
	switch choice {
	case "", "none", ToolChoiceAuto:
		return nil
	case ToolChoiceToolsRequired:
		return InvalidToolChoiceErr("Z.AI does not support required tool choice")
	default:
		return InvalidToolChoiceErr("Z.AI does not support named tool choice")
	}
}

func (g *ZaiGenerator) buildTextRequest(model string, dialog Dialog, instructions string, tools []zai.FunctionToolSchema, options *zaiGenerationOptions, stream bool) (zai.ChatCompletionTextRequest, error) {
	messages, err := g.buildTextMessages(dialog, instructions)
	if err != nil {
		return zai.ChatCompletionTextRequest{}, err
	}

	request := zai.ChatCompletionTextRequest{
		Model:    zai.ChatCompletionTextRequestModel(model),
		Messages: messages,
		Stream:   zai.NewOptBool(stream),
		Thinking: zaiThinking(options),
	}
	if options.ThinkingBudget != "" {
		effort := zai.ChatCompletionTextRequestReasoningEffort(options.ThinkingBudget)
		if err := effort.Validate(); err != nil {
			return zai.ChatCompletionTextRequest{}, &InvalidParameterErr{Parameter: GenerationOptionThinkingBudget, Reason: err.Error()}
		}
		request.ReasoningEffort = zai.NewOptChatCompletionTextRequestReasoningEffort(effort)
	}
	includeTools := applyZaiTextOptions(&request, options)
	if includeTools && len(tools) > 0 {
		for _, tool := range tools {
			request.Tools = append(request.Tools, zai.NewFunctionToolSchemaChatCompletionTextRequestToolsItem(tool))
		}
		if stream {
			request.ToolStream = zai.NewOptBool(true)
		}
	}
	return request, nil
}

func (g *ZaiGenerator) buildVisionRequest(model string, dialog Dialog, instructions string, tools []zai.FunctionToolSchema, options *zaiGenerationOptions, stream bool) (zai.ChatCompletionVisionRequest, error) {
	messages, err := g.buildVisionMessages(dialog, instructions)
	if err != nil {
		return zai.ChatCompletionVisionRequest{}, err
	}

	request := zai.ChatCompletionVisionRequest{
		Model:    zai.ChatCompletionVisionRequestModel(model),
		Messages: messages,
		Stream:   zai.NewOptBool(stream),
		Thinking: zaiThinking(options),
	}
	if options.ThinkingBudget != "" {
		effort := zai.ChatCompletionVisionRequestReasoningEffort(options.ThinkingBudget)
		if err := effort.Validate(); err != nil {
			return zai.ChatCompletionVisionRequest{}, &InvalidParameterErr{Parameter: GenerationOptionThinkingBudget, Reason: err.Error()}
		}
		request.ReasoningEffort = zai.NewOptChatCompletionVisionRequestReasoningEffort(effort)
	}
	includeTools := applyZaiVisionOptions(&request, options)
	if includeTools && len(tools) > 0 {
		for _, tool := range tools {
			request.Tools = append(request.Tools, zai.NewFunctionToolSchemaChatCompletionVisionRequestToolsItem(tool))
		}
	}
	return request, nil
}

func zaiThinking(options *zaiGenerationOptions) zai.OptChatThinking {
	thinkingType := zai.ChatThinkingTypeEnabled
	if !options.ThinkingEnabled {
		thinkingType = zai.ChatThinkingTypeDisabled
	}
	return zai.NewOptChatThinking(zai.ChatThinking{
		Type:          zai.NewOptChatThinkingType(thinkingType),
		ClearThinking: zai.NewOptBool(options.ClearThinking),
	})
}

func applyZaiTextOptions(request *zai.ChatCompletionTextRequest, options *zaiGenerationOptions) bool {
	includeTools := true
	if options.Temperature != nil {
		request.Temperature = zai.NewOptFloat32(float32(*options.Temperature))
	}
	if options.TopP != nil {
		request.TopP = zai.NewOptFloat32(float32(*options.TopP))
	}
	if options.MaxGenerationTokens != nil {
		request.MaxTokens = zai.NewOptInt(*options.MaxGenerationTokens)
	}
	if len(options.StopSequences) > 0 {
		request.Stop = options.StopSequences
	}
	if options.ToolChoice != "" {
		switch options.ToolChoice {
		case "none":
			includeTools = false
		case ToolChoiceAuto:
			request.ToolChoice = zai.NewOptChatCompletionTextRequestToolChoice(zai.ChatCompletionTextRequestToolChoiceAuto)
		}
	}
	return includeTools
}

func applyZaiVisionOptions(request *zai.ChatCompletionVisionRequest, options *zaiGenerationOptions) bool {
	includeTools := true
	if options.Temperature != nil {
		request.Temperature = zai.NewOptFloat32(float32(*options.Temperature))
	}
	if options.TopP != nil {
		request.TopP = zai.NewOptFloat32(float32(*options.TopP))
	}
	if options.MaxGenerationTokens != nil {
		request.MaxTokens = zai.NewOptInt(*options.MaxGenerationTokens)
	}
	if len(options.StopSequences) > 0 {
		request.Stop = options.StopSequences
	}
	if options.ToolChoice != "" {
		switch options.ToolChoice {
		case "none":
			includeTools = false
		case ToolChoiceAuto:
			request.ToolChoice = zai.NewOptChatCompletionVisionRequestToolChoice(zai.ChatCompletionVisionRequestToolChoiceAuto)
		}
	}
	return includeTools
}

func zaiResponseExtraFields(id, requestID, model zai.OptString, created zai.OptInt) map[string]interface{} {
	extraFields := make(map[string]interface{})
	if value, ok := id.Get(); ok {
		extraFields[ZaiResponseExtraFieldID] = value
	}
	if value, ok := requestID.Get(); ok {
		extraFields[ZaiResponseExtraFieldRequestID] = value
	}
	if value, ok := created.Get(); ok {
		extraFields[ZaiResponseExtraFieldCreated] = value
	}
	if value, ok := model.Get(); ok {
		extraFields[ZaiResponseExtraFieldModel] = value
	}
	return extraFields
}

func zaiWebSearchResults(results []zai.WebSearchObjectResponse) []map[string]interface{} {
	converted := make([]map[string]interface{}, 0, len(results))
	for _, result := range results {
		value := make(map[string]interface{})
		if field, ok := result.Title.Get(); ok {
			value["title"] = field
		}
		if field, ok := result.Content.Get(); ok {
			value["content"] = field
		}
		if field, ok := result.Link.Get(); ok {
			value["link"] = field
		}
		if field, ok := result.Media.Get(); ok {
			value["media"] = field
		}
		if field, ok := result.Icon.Get(); ok {
			value["icon"] = field
		}
		if field, ok := result.Refer.Get(); ok {
			value["refer"] = field
		}
		if field, ok := result.PublishDate.Get(); ok {
			value["publish_date"] = field
		}
		converted = append(converted, value)
	}
	return converted
}

// Generate sends one Z.AI Chat Completions request and normalizes text,
// reasoning, tool calls, hosted-search results, usage, and provider failures
// into [Response].
func (g *ZaiGenerator) Generate(ctx context.Context, generationRequest GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("zai: client not initialized")
	}
	if len(generationRequest.Dialog) == 0 {
		return Response{}, ErrEmptyDialog
	}

	request, params, err := g.buildRequest(generationRequest, false)
	if err != nil {
		return Response{}, err
	}

	rawResponse, err := g.client.PaasV4ChatCompletionsPost(ctx, request, params)
	if err != nil {
		return Response{}, mapZAIError(err)
	}
	resp, ok := rawResponse.(*zai.ChatCompletionResponse)
	if !ok {
		if stream, isStream := rawResponse.(*zai.PaasV4ChatCompletionsPostOKTextEventStream); isStream {
			_ = stream.Close()
		}
		return Response{}, fmt.Errorf("zai: expected JSON completion response, got %T", rawResponse)
	}

	result := Response{
		UsageMetadata: make(Metadata),
		ExtraFields:   zaiResponseExtraFields(resp.ID, resp.RequestID, resp.Model, resp.Created),
	}
	if len(resp.WebSearch) > 0 {
		result.ExtraFields[ZaiResponseExtraFieldWebSearchResults] = zaiWebSearchResults(resp.WebSearch)
	}
	if usage, ok := resp.Usage.Get(); ok {
		var cachedTokens float64
		if details, ok := usage.PromptTokensDetails.Get(); ok {
			cachedTokens = optFloat64(details.CachedTokens)
		}
		addZaiUsageMetadata(result.UsageMetadata, zaiUsage{
			PromptTokens:     optFloat64(usage.PromptTokens),
			CompletionTokens: optFloat64(usage.CompletionTokens),
			CachedTokens:     cachedTokens,
		})
	}

	var hasToolCalls bool
	for _, choice := range resp.Choices {
		var blocks []Block
		if msg, ok := choice.Message.Get(); ok {
			if rc, ok := msg.ReasoningContent.Get(); ok && rc != "" {
				blocks = append(blocks, zaiThinkingBlock(rc))
			}
			if content, ok := msg.Content.Get(); ok && content != "" {
				blocks = append(blocks, Block{
					BlockType:    Content,
					ModalityType: Text,
					MimeType:     "text/plain",
					Content:      Str(content),
				})
			}
			if len(msg.ToolCalls) > 0 {
				hasToolCalls = true
				for _, tc := range msg.ToolCalls {
					block, err := zaiToolCallBlockFromResponse(tc)
					if err != nil {
						return result, err
					}
					blocks = append(blocks, block)
				}
			}
		}
		result.Candidates = append(result.Candidates, Message{Role: Assistant, Blocks: blocks})
	}

	if len(resp.Choices) > 0 {
		result.FinishReason, err = zaiFinishReason(resp.Choices[0].FinishReason.Or(""))
		if err != nil {
			return result, err
		}
	}
	if hasToolCalls && result.FinishReason == EndTurn {
		result.FinishReason = ToolUse
	}
	return result, nil
}

// Stream starts one Z.AI SSE stream when iterated and emits ordered reasoning,
// text, tool-call, metadata, and terminal-error [StreamChunk] values.
func (g *ZaiGenerator) Stream(ctx context.Context, generationRequest GenerationRequest) iter.Seq[StreamChunk] {
	return func(yield func(StreamChunk) bool) {
		if g.client == nil {
			yield(StreamChunk{Err: fmt.Errorf("zai: client not initialized")})
			return
		}
		if len(generationRequest.Dialog) == 0 {
			yield(StreamChunk{Err: ErrEmptyDialog})
			return
		}

		request, params, err := g.buildRequest(generationRequest, true)
		if err != nil {
			yield(StreamChunk{Err: err})
			return
		}

		rawResponse, err := g.client.PaasV4ChatCompletionsPost(ctx, request, params)
		if err != nil {
			yield(StreamChunk{Err: mapZAIError(err)})
			return
		}
		stream, ok := rawResponse.(*zai.PaasV4ChatCompletionsPostOKTextEventStream)
		if !ok {
			yield(StreamChunk{Err: fmt.Errorf("zai: expected event stream response, got %T", rawResponse)})
			return
		}
		defer stream.Close()

		var finalUsage zai.ChatCompletionStreamUsage
		var hasFinalUsage bool
		responseExtraFields := make(map[string]interface{})
		for {
			event, err := stream.Next(ctx)
			if err != nil {
				yield(StreamChunk{Err: mapZAIError(err)})
				return
			}
			if event.Data.IsPaasV4ChatCompletionsPostOKTextEventStreamEventData1() {
				break
			}
			chunk, ok := event.Data.GetChatCompletionStreamResponse()
			if !ok {
				yield(StreamChunk{Err: fmt.Errorf("zai: unexpected event data type %q", event.Data.Type)})
				return
			}
			if usage, ok := chunk.Usage.Get(); ok {
				finalUsage = usage
				hasFinalUsage = true
			}
			chunkExtraFields := zaiResponseExtraFields(chunk.ID, chunk.RequestID, chunk.Model, chunk.Created)
			maps.Copy(responseExtraFields, chunkExtraFields)
			for _, choice := range chunk.Choices {
				if finishReason := choice.FinishReason.Or(""); finishReason != "" {
					if _, finishErr := zaiFinishReason(finishReason); finishErr != nil {
						yield(StreamChunk{Err: finishErr})
						return
					}
				}

				if refusal := choice.Delta.Refusal.Or(""); refusal != "" {
					yield(StreamChunk{Err: ContentPolicyErr("content refused")})
					return
				}

				if reasoning := choice.Delta.ReasoningContent.Or(""); reasoning != "" {
					if !yield(StreamChunk{
						Block:               zaiThinkingBlock(reasoning),
						ResponseExtraFields: chunkExtraFields,
						CandidatesIndex:     choice.Index,
					}) {
						return
					}
				}
				if content := choice.Delta.Content.Or(""); content != "" {
					if !yield(StreamChunk{
						Block: Block{
							BlockType:    Content,
							ModalityType: Text,
							MimeType:     "text/plain",
							Content:      Str(content),
						},
						ResponseExtraFields: chunkExtraFields,
						CandidatesIndex:     choice.Index,
					}) {
						return
					}
				}
				for _, tc := range choice.Delta.ToolCalls {
					if name := tc.Function.Name.Or(""); name != "" {
						if !yield(StreamChunk{
							Block: Block{
								ID:           tc.ID.Or(""),
								BlockType:    ToolCall,
								ModalityType: Text,
								MimeType:     "text/plain",
								Content:      Str(name),
							},
							ResponseExtraFields: chunkExtraFields,
							CandidatesIndex:     choice.Index,
						}) {
							return
						}
					}
					if arguments := tc.Function.Arguments.Or(""); arguments != "" {
						if !yield(StreamChunk{
							Block: Block{
								BlockType:    ToolCall,
								ModalityType: Text,
								MimeType:     "text/plain",
								Content:      Str(arguments),
							},
							ResponseExtraFields: chunkExtraFields,
							CandidatesIndex:     choice.Index,
						}) {
							return
						}
					}
				}
			}
		}

		metadata := make(Metadata)
		if hasFinalUsage {
			var cachedTokens float64
			if details, ok := finalUsage.PromptTokensDetails.Get(); ok {
				cachedTokens = optFloat64(details.CachedTokens)
			}
			addZaiUsageMetadata(metadata, zaiUsage{
				PromptTokens:     optFloat64(finalUsage.PromptTokens),
				CompletionTokens: optFloat64(finalUsage.CompletionTokens),
				CachedTokens:     cachedTokens,
			})
		}
		terminalBlock := SeparatorBlock()
		if len(metadata) > 0 {
			terminalBlock = MetadataBlock(metadata)
		}
		if len(metadata) > 0 || len(responseExtraFields) > 0 {
			yield(StreamChunk{
				Block:               terminalBlock,
				ResponseExtraFields: responseExtraFields,
				CandidatesIndex:     0,
			})
		}
	}
}

func zaiFinishReason(reason string) (FinishReason, error) {
	switch reason {
	case "stop":
		return EndTurn, nil
	case "tool_calls":
		return ToolUse, nil
	case "length", "model_context_window_exceeded":
		return MaxGenerationLimit, ErrMaxGenerationLimit
	case "content_filter":
		return ContentPolicyViolation, ContentPolicyErr("content filtered")
	case "sensitive":
		return ContentPolicyViolation, ContentPolicyErr("content flagged as sensitive")
	case "network_error":
		return Unknown, &ApiErr{
			Provider: ProviderZAI,
			Kind:     APIErrorKindServiceUnavailable,
			Message:  "generation stopped because Z.AI reported a network error",
		}
	default:
		return Unknown, nil
	}
}

type zaiUsage struct {
	PromptTokens     float64
	CompletionTokens float64
	CachedTokens     float64
}

func addZaiUsageMetadata(metadata Metadata, usage zaiUsage) {
	if usage.PromptTokens > 0 {
		metadata[UsageMetricInputTokens] = int(usage.PromptTokens)
	}
	if usage.CompletionTokens > 0 {
		metadata[UsageMetricGenerationTokens] = int(usage.CompletionTokens)
	}
	if usage.CachedTokens > 0 {
		metadata[UsageMetricCacheReadTokens] = int(usage.CachedTokens)
	}
}

func optFloat64(v zai.OptFloat64) float64 {
	if f, ok := v.Get(); ok {
		return f
	}
	return 0
}

func zaiThinkingBlock(content string) Block {
	return Block{
		BlockType:    Thinking,
		ModalityType: Text,
		MimeType:     "text/plain",
		Content:      Str(content),
		ExtraFields: map[string]interface{}{
			ThinkingExtraFieldGeneratorKey: ThinkingGeneratorZai,
		},
	}
}

func zaiToolCallBlockFromResponse(tc zai.ChatCompletionResponseMessageToolCall) (Block, error) {
	id := tc.ID.Or("")
	var name string
	var arguments string
	if fn, ok := tc.Function.Get(); ok {
		name = fn.Name
		arguments = fn.Arguments
	}
	params := map[string]any{}
	if strings.TrimSpace(arguments) != "" {
		if err := json.Unmarshal([]byte(arguments), &params); err != nil {
			return Block{}, err
		}
	}
	content, err := json.Marshal(ToolCallInput{Name: name, Parameters: params})
	if err != nil {
		return Block{}, err
	}
	return Block{
		ID:           id,
		BlockType:    ToolCall,
		ModalityType: Text,
		MimeType:     "application/json",
		Content:      Str(content),
	}, nil
}

func decodeZaiToolCallArguments(raw json.RawMessage, params *map[string]any) error {
	if bytes.Equal(raw, []byte("null")) || len(raw) == 0 {
		return nil
	}
	if raw[0] == '"' {
		var args string
		if err := json.Unmarshal(raw, &args); err != nil {
			return err
		}
		if strings.TrimSpace(args) == "" {
			return nil
		}
		return json.Unmarshal([]byte(args), params)
	}
	return json.Unmarshal(raw, params)
}

// Count sends the model, instructions, dialog, tools, and multimodal input to
// Z.AI's tokenizer endpoint. It uses the same message conversion as generation,
// returns total input tokens, and honors context cancellation.
func (g *ZaiGenerator) Count(ctx context.Context, request GenerationRequest) (uint, error) {
	// Count builds the matching text or vision request before calling Z.AI.
	if g.client == nil {
		return 0, fmt.Errorf("zai: client not initialized")
	}
	if len(request.Dialog) == 0 {
		return 0, ErrEmptyDialog
	}

	instructions, err := joinedTextInstructions(request.Instructions)
	if err != nil {
		return 0, err
	}
	tools, err := convertToolsToZai(request.Tools)
	if err != nil {
		return 0, err
	}

	var tokenizerRequest zai.PaasV4TokenizerPostReq
	if g.dialogNeedsVision(request.Dialog) {
		messages, err := g.buildVisionMessages(request.Dialog, instructions)
		if err != nil {
			return 0, fmt.Errorf("zai: prepare token counting messages: %w", err)
		}
		providerRequest := zai.ChatCompletionVisionRequest{
			Model:    zai.ChatCompletionVisionRequestModel(request.Model),
			Messages: messages,
		}
		for _, tool := range tools {
			providerRequest.Tools = append(providerRequest.Tools, zai.NewFunctionToolSchemaChatCompletionVisionRequestToolsItem(tool))
		}
		tokenizerRequest = zai.NewChatCompletionVisionRequestPaasV4TokenizerPostReq(providerRequest)
	} else {
		messages, err := g.buildTextMessages(request.Dialog, instructions)
		if err != nil {
			return 0, fmt.Errorf("zai: prepare token counting messages: %w", err)
		}
		providerRequest := zai.ChatCompletionTextRequest{
			Model:    zai.ChatCompletionTextRequestModel(request.Model),
			Messages: messages,
		}
		for _, tool := range tools {
			providerRequest.Tools = append(providerRequest.Tools, zai.NewFunctionToolSchemaChatCompletionTextRequestToolsItem(tool))
		}
		tokenizerRequest = zai.NewChatCompletionTextRequestPaasV4TokenizerPostReq(providerRequest)
	}

	response, err := g.client.PaasV4TokenizerPost(ctx, tokenizerRequest)
	if err != nil {
		return 0, mapZAIError(err)
	}
	total, ok := response.Usage.TotalTokens.Get()
	if !ok {
		return 0, fmt.Errorf("zai: token counting response missing total_tokens")
	}
	if math.IsNaN(total) || math.IsInf(total, 0) || total < 0 || math.Trunc(total) != total || total >= float64(^uint(0)) {
		return 0, fmt.Errorf("zai: token counting response has invalid total_tokens %v", total)
	}
	return uint(total), nil
}

func mapZAIError(err error) error {
	var statusErr *zai.ErrorResponseStatusCode
	if !errors.As(err, &statusErr) {
		return err
	}
	rawBody, _ := json.Marshal(statusErr.Response)
	message := ""
	if detail, ok := statusErr.Response.GetError(); ok {
		message = detail.Message
	} else if envelope, ok := statusErr.Response.GetErrorEnvelope(); ok {
		message = envelope.Error.Message
	}
	return &ApiErr{
		Provider:   ProviderZAI,
		Kind:       classifyHTTPStatus(statusErr.StatusCode),
		StatusCode: statusErr.StatusCode,
		Message:    message,
		RawBody:    string(rawBody),
		Cause:      err,
	}
}

var _ Generator = (*ZaiGenerator)(nil)
var _ StreamingGenerator = (*ZaiGenerator)(nil)
var _ TokenCounter = (*ZaiGenerator)(nil)
