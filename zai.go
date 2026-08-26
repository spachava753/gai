package gai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"os"
	"strings"

	"github.com/go-faster/jx"

	"github.com/spachava753/gai/internal/zai"
)

// ZaiGenerator implements the Generator and StreamingGenerator interfaces for Z.AI API.
// Z.AI provides OpenAI-compatible endpoints with extended thinking/reasoning capabilities.
//
// Key features:
//   - OpenAI-compatible chat completions API
//   - Interleaved thinking: the model can reason between tool calls
//   - Preserved thinking: reasoning context can be retained across turns
//   - Streaming with Server-Sent Events (SSE)
//
// Supported models include glm-5.1, glm-5, glm-4.7, glm-4.6, glm-4.5, and variants.
type ZaiGenerator struct {
	client ZaiCompletionService
}

// ZaiCompletionService defines the generated Z.AI chat completions client surface.
type ZaiCompletionService interface {
	PaasV4ChatCompletionsPost(ctx context.Context, request zai.PaasV4ChatCompletionsPostReq, params zai.PaasV4ChatCompletionsPostParams) (zai.PaasV4ChatCompletionsPostRes, error)
}

const (
	ZaiGenerationOptionThinkingEnabled = "zai_thinking_enabled"
	ZaiGenerationOptionClearThinking   = "zai_clear_thinking"
)

// WithZaiThinking controls thinking mode for one generation request.
func WithZaiThinking(enabled bool) GenerationOption {
	return func(options GenerationOptions) {
		options[ZaiGenerationOptionThinkingEnabled] = enabled
	}
}

// WithZaiClearThinking controls whether reasoning content from earlier turns is cleared.
func WithZaiClearThinking(clear bool) GenerationOption {
	return func(options GenerationOptions) {
		options[ZaiGenerationOptionClearThinking] = clear
	}
}

const (
	zaiBaseURL = "https://api.z.ai/api"

	// ZaiExtraFieldURL can be set on Image or Video content blocks to pass a remote URL.
	// For PDFs, set this to a PDF URL on a PDFBlock; Z.AI file inputs require URLs.
	ZaiExtraFieldURL = "zai_url"
)

// NewZaiGenerator creates a stateless Z.AI generator using the generated client.
// If client is nil, a generated client is created with the Z.AI base URL.
// apiKey is read from Z_API_KEY environment variable if empty.
func NewZaiGenerator(client ZaiCompletionService, apiKey string) *ZaiGenerator {
	if apiKey == "" {
		apiKey = os.Getenv("Z_API_KEY")
	}

	if client == nil {
		client, _ = newDefaultZaiClient(apiKey)
	}
	return &ZaiGenerator{client: client}
}

type zaiSecuritySource struct {
	apiKey string
}

func (s zaiSecuritySource) BearerAuth(ctx context.Context, operationName zai.OperationName) (zai.BearerAuth, error) {
	return zai.BearerAuth{Token: s.apiKey}, nil
}

func newDefaultZaiClient(apiKey string) (*zai.Client, error) {
	return zai.NewClient(zaiBaseURL, zaiSecuritySource{apiKey: apiKey})
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
	ThinkingEnabled     bool
	ClearThinking       bool
}

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
	if isZaiURL(content) {
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

func isZaiURL(s string) bool {
	return strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://") || strings.HasPrefix(s, "data:")
}

func isZaiRemoteURL(s string) bool {
	return strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://")
}

func (g *ZaiGenerator) buildRequest(generationRequest GenerationRequest, stream bool) (zai.PaasV4ChatCompletionsPostReq, zai.PaasV4ChatCompletionsPostParams, error) {
	options, err := parseZaiGenerationOptions(generationRequest.Options)
	if err != nil {
		return zai.PaasV4ChatCompletionsPostReq{}, zai.PaasV4ChatCompletionsPostParams{}, err
	}
	if err := validateZaiOutputModalities(options); err != nil {
		return zai.PaasV4ChatCompletionsPostReq{}, zai.PaasV4ChatCompletionsPostParams{}, err
	}
	tools, err := convertToolsToZai(generationRequest.Tools)
	if err != nil {
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

func validateZaiOutputModalities(options *zaiGenerationOptions) error {
	for _, m := range options.OutputModalities {
		if m != Text {
			return UnsupportedOutputModalityErr(m.String())
		}
	}
	return nil
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
		case ToolChoiceAuto, ToolChoiceToolsRequired:
			request.ToolChoice = zai.NewOptChatCompletionTextRequestToolChoice(zai.ChatCompletionTextRequestToolChoiceAuto)
		default:
			// The generated Z.AI schema currently only permits "auto" for tool_choice.
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
		case ToolChoiceAuto, ToolChoiceToolsRequired:
			request.ToolChoice = zai.NewOptChatCompletionVisionRequestToolChoice(zai.ChatCompletionVisionRequestToolChoiceAuto)
		default:
			request.ToolChoice = zai.NewOptChatCompletionVisionRequestToolChoice(zai.ChatCompletionVisionRequestToolChoiceAuto)
		}
	}
	return includeTools
}

// Generate implements Generator
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

	result := Response{UsageMetadata: make(Metadata)}
	if usage, ok := resp.Usage.Get(); ok {
		addZaiUsageMetadata(result.UsageMetadata, zaiUsage{
			PromptTokens:     optFloat64(usage.PromptTokens),
			CompletionTokens: optFloat64(usage.CompletionTokens),
			CachedTokens:     optCachedTokens(usage.PromptTokensDetails),
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
		reason := resp.Choices[0].FinishReason.Or("")
		switch reason {
		case "stop":
			result.FinishReason = EndTurn
		case "length", "model_context_window_exceeded":
			result.FinishReason = MaxGenerationLimit
			return result, ErrMaxGenerationLimit
		case "tool_calls":
			result.FinishReason = ToolUse
		case "content_filter":
			result.FinishReason = ContentPolicyViolation
			return result, ContentPolicyErr("content filtered")
		case "sensitive":
			result.FinishReason = ContentPolicyViolation
			return result, ContentPolicyErr("content flagged as sensitive")
		default:
			result.FinishReason = Unknown
		}
	}
	if hasToolCalls && result.FinishReason == EndTurn {
		result.FinishReason = ToolUse
	}
	return result, nil
}

// Stream implements StreamingGenerator
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
			for _, choice := range chunk.Choices {
				finishReason := choice.FinishReason.Or("")
				switch finishReason {
				case "length", "model_context_window_exceeded":
					yield(StreamChunk{Err: ErrMaxGenerationLimit})
					return
				case "content_filter", "sensitive":
					yield(StreamChunk{Err: ContentPolicyErr("content filtered")})
					return
				}

				if refusal := choice.Delta.Refusal.Or(""); refusal != "" {
					yield(StreamChunk{Err: ContentPolicyErr("content refused")})
					return
				}

				if reasoning := choice.Delta.ReasoningContent.Or(""); reasoning != "" {
					if !yield(StreamChunk{Block: zaiThinkingBlock(reasoning), CandidatesIndex: choice.Index}) {
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
						CandidatesIndex: choice.Index,
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
							CandidatesIndex: choice.Index,
						}) {
							return
						}
					}
					if arguments := tc.Function.Arguments.Or(""); arguments != "" {
						if !yield(StreamChunk{
							Block: Block{
								ID:           tc.ID.Or(""),
								BlockType:    ToolCall,
								ModalityType: Text,
								MimeType:     "text/plain",
								Content:      Str(arguments),
							},
							CandidatesIndex: choice.Index,
						}) {
							return
						}
					}
				}
			}
		}

		if hasFinalUsage {
			var cachedTokens float64
			if details, ok := finalUsage.PromptTokensDetails.Get(); ok {
				cachedTokens = optFloat64(details.CachedTokens)
			}
			metadata := make(Metadata)
			addZaiUsageMetadata(metadata, zaiUsage{
				PromptTokens:     optFloat64(finalUsage.PromptTokens),
				CompletionTokens: optFloat64(finalUsage.CompletionTokens),
				CachedTokens:     cachedTokens,
			})
			if len(metadata) > 0 {
				yield(StreamChunk{Block: MetadataBlock(metadata), CandidatesIndex: 0})
			}
		}
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

func optCachedTokens(v zai.OptChatCompletionResponseUsagePromptTokensDetails) float64 {
	if details, ok := v.Get(); ok {
		return optFloat64(details.CachedTokens)
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

func mapZAIError(err error) error {
	var statusErr *zai.ErrorStatusCode
	if !errors.As(err, &statusErr) {
		return err
	}
	rawBody, _ := json.Marshal(statusErr.Response)
	return &ApiErr{
		Provider:   ProviderZAI,
		Kind:       classifyHTTPStatus(statusErr.StatusCode),
		StatusCode: statusErr.StatusCode,
		Message:    statusErr.Response.Message,
		RawBody:    string(rawBody),
		Cause:      err,
	}
}

var _ Generator = (*ZaiGenerator)(nil)
var _ StreamingGenerator = (*ZaiGenerator)(nil)
