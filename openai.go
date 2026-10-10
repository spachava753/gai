package gai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"  // Register GIF format
	_ "image/jpeg" // Register JPEG format
	_ "image/png"  // Register PNG format
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"

	"github.com/oapi-codegen/nullable"
	"github.com/pkoukk/tiktoken-go"

	wire "github.com/spachava753/gai/internal/openai"
)

const (
	// OpenAIExtraFieldImageWidth is the int pixel-width key in
	// [Block.ExtraFields]. [OpenAiGenerator.Count] uses it when dimensions cannot
	// be decoded from image data.
	OpenAIExtraFieldImageWidth = "width"

	// OpenAIExtraFieldImageHeight is the int pixel-height key in
	// [Block.ExtraFields]. [OpenAiGenerator.Count] uses it with
	// [OpenAIExtraFieldImageWidth].
	OpenAIExtraFieldImageHeight = "height"

	// OpenAIExtraFieldImageDetail is the string [Block.ExtraFields] key for
	// OpenAI image detail. [OpenAiGenerator.Count] recognizes "low" and "high"
	// and defaults an absent or unrecognized value to "high".
	OpenAIExtraFieldImageDetail = "detail"
)

func init() {
	// The tiktoken library does not have the update-to-date mappings of model names to encodings,
	// so we manually add them here.
	tiktoken.MODEL_TO_ENCODING["o3"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["o4-mini"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1-mini"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1-nano"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1-2025-04-14"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1-mini-2025-04-14"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4.1-nano--2025-04-14"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["o3-mini"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["o3-mini-2025-01-31"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["o1"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["o1-2024-12-17"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["o1-preview"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["o1-preview-2024-09-12"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["o1-mini"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["o1-mini-2024-09-12"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4o"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4o-2024-11-20"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4o-2024-08-06"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4o-2024-05-13"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4o-audio-preview"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4o-audio-preview-2024-10-01"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4o-audio-preview-2024-12-17"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4o-mini-audio-preview"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4o-mini-audio-preview-2024-12-17"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["chatgpt-4o-latest"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4o-mini"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4o-mini-2024-07-18"] = tiktoken.MODEL_O200K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4-turbo"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4-turbo-2024-04-09"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4-0125-preview"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4-turbo-preview"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4-1106-preview"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4-vision-preview"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4-0314"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4-0613"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4-32k"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4-32k-0314"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-4-32k-0613"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-3.5-turbo"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-3.5-turbo-16k"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-3.5-turbo-0301"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-3.5-turbo-0613"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-3.5-turbo-1106"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-3.5-turbo-0125"] = tiktoken.MODEL_CL100K_BASE
	tiktoken.MODEL_TO_ENCODING["gpt-3.5-turbo-16k-0613"] = tiktoken.MODEL_CL100K_BASE
}

// OpenAIDefaultBaseURL is used by NewOpenAiGenerator when baseURL is empty.
const OpenAIDefaultBaseURL = "https://api.openai.com/v1"

const (
	// OpenAIExtraFieldWireFields holds map[string]json.RawMessage provider fields
	// on Message.ExtraFields or Block.ExtraFields, at their original wire scope.
	// Conversion owns content, role, and tool identity; these fields cannot replace
	// them. Returned maps belong to the response, not the generator.
	OpenAIExtraFieldWireFields = "openai_wire_fields"
	// OpenAIExtraFieldChoice holds map[string]json.RawMessage choice metadata
	// (including finish reason, index, and logprobs) in Message.ExtraFields.
	// It is retained for inspection, not replayed as request-message fields.
	OpenAIExtraFieldChoice = "openai_choice"
	// OpenAIExtraFieldAudio holds map[string]json.RawMessage output audio
	// metadata in Block.ExtraFields, excluding its ID and binary data. Replay
	// sends the block's audio ID rather than the output-only metadata.
	OpenAIExtraFieldAudio = "openai_audio"
	// OpenAIExtraFieldReasoningSource records which response field supplied a
	// [Thinking] block's text. It is stored in [Block.ExtraFields] as a string:
	// "reasoning_details", "reasoning_content", or "reasoning".
	//
	// The generator sets this automatically. Keep it and the returned message's
	// [OpenAIExtraFieldWireFields] when adding an assistant response to a dialog.
	// This lets the next request send the provider's original reasoning data
	// rather than another copy made from the displayed text. Editing only the
	// block's text does not change the original data sent back to the provider.
	OpenAIExtraFieldReasoningSource = "openai_reasoning_source"
	// OpenAIResponseExtraFieldHeaders holds a cloned http.Header in Response.ExtraFields.
	OpenAIResponseExtraFieldHeaders = "openai_headers"
	// OpenAIResponseExtraFieldWireFields holds response-level map[string]json.RawMessage
	// metadata in Response.ExtraFields, excluding choices and usage.
	OpenAIResponseExtraFieldWireFields = "openai_wire_fields"
	// OpenAIResponseExtraFieldUsage retains the complete provider usage object
	// as map[string]json.RawMessage in Response.ExtraFields, including token
	// breakdowns not represented by the common UsageMetadata metrics.
	OpenAIResponseExtraFieldUsage = "openai_usage"
	// OpenAIGenerationOptionTokenLimitField selects the wire field for
	// WithMaxGenerationTokens. See WithOpenAITokenLimitField.
	OpenAIGenerationOptionTokenLimitField = "openai_token_limit_field"
	// OpenAIGenerationOptionExtraBody is the map[string]json.RawMessage request
	// field map set by [WithOpenAIExtraBody].
	OpenAIGenerationOptionExtraBody = "openai_extra_body"
	// OpenAIGenerationOptionStreamUsage is the bool set by [WithOpenAIStreamUsage].
	OpenAIGenerationOptionStreamUsage = "openai_stream_usage"
)

// WithOpenAITokenLimitField selects "max_tokens" or "max_completion_tokens" for
// WithMaxGenerationTokens. The default is "max_completion_tokens", preserving
// OpenAiGenerator's OpenAI behavior. Selection is never inferred from the model.
func WithOpenAITokenLimitField(field string) GenerationOption {
	return func(options GenerationOptions) { options[OpenAIGenerationOptionTokenLimitField] = field }
}

// WithOpenAIExtraBody supplies native JSON request fields such as thinking,
// tool_stream, or response_format. Values override corresponding common options;
// explicit JSON null is preserved. Model, messages, tools, and stream remain
// owned by the generation request and cannot be overridden here. The map is
// borrowed until generation; each call copies it without mutating caller data.
// No provider-specific defaults or capability decisions are inferred.
func WithOpenAIExtraBody(fields map[string]json.RawMessage) GenerationOption {
	return func(options GenerationOptions) { options[OpenAIGenerationOptionExtraBody] = fields }
}

// WithOpenAIStreamUsage controls whether Stream sends stream_options.include_usage.
// The default is true. False omits stream_options for APIs that report usage
// without this switch; it does not discard usage received from the provider.
// WithOpenAIExtraBody can supply an explicit stream_options object instead.
func WithOpenAIStreamUsage(enabled bool) GenerationOption {
	return func(options GenerationOptions) { options[OpenAIGenerationOptionStreamUsage] = enabled }
}

// OpenAiGenerator sends Chat Completions requests to OpenAI or a compatible
// endpoint. Pass the model, conversation, tools, and options in each
// [GenerationRequest]. Supported inputs include text, images, PDFs, and audio;
// supported outputs include text, audio, and function calls. Actual support
// depends on the endpoint and model. It implements [Generator],
// [StreamingGenerator], and [TokenCounter]. [System] messages in
// [GenerationRequest.Dialog] retain their position after any initial Instructions;
// endpoint and model support are validated by the server.
//
// To display model reasoning, read the Content of blocks whose BlockType is
// [Thinking]. The generator creates these blocks from reasoning_content,
// reasoning, and readable text or summaries in reasoning_details. Encrypted
// reasoning and summaries that are not strings are kept as metadata, not shown
// as thinking text.
//
// A provider may return the same reasoning in more than one field. Identical
// text is shown once. If a later field starts with text already sent, Stream
// emits only the added text. If the fields contain different text, both are
// kept. Streaming text follows the order it arrives, which can differ from
// the order returned by Generate.
//
// To continue the conversation, append the returned assistant [Message] to
// [GenerationRequest.Dialog], keeping its message and block ExtraFields.
// These fields preserve the original reasoning, signatures, and other data
// the provider may need on the next turn. You do not need to parse or rebuild
// reasoning_details yourself. See [OpenAIExtraFieldReasoningSource].
//
// Use [WithOpenAIExtraBody] for native request fields and [WithOpenAIStreamUsage]
// to control whether streaming requests ask for usage. The full usage object is
// available under [OpenAIResponseExtraFieldUsage]. [WithMaxGenerationTokens]
// sends max_completion_tokens by default; [WithOpenAITokenLimitField] can select
// max_tokens instead.
//
// Stream does not reconnect or retry. It continues reading after [DONE] until
// the server closes the stream, so it can retain trailing metadata. Supply a
// context deadline to bound the wait.
type OpenAiGenerator struct{ client *wire.Client }

// NewOpenAiGenerator constructs a stateless Chat Completions adapter. A nil
// httpClient uses http.DefaultClient; an empty baseURL uses OpenAIDefaultBaseURL.
// Credentials are explicit: an empty apiKey returns ErrMissingAPIKey. The base
// URL includes the API prefix (such as /v1), not /chat/completions. No SDK retry
// policy or environment configuration is applied.
func NewOpenAiGenerator(httpClient *http.Client, baseURL, apiKey string) (*OpenAiGenerator, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("openai: %w", ErrMissingAPIKey)
	}
	if baseURL == "" {
		baseURL = OpenAIDefaultBaseURL
	}
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "https" && parsed.Scheme != "http") || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, InvalidParameterErr{Parameter: "baseURL", Reason: "expected an HTTP(S) base URL without query or fragment"}
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	client, err := wire.NewClient(baseURL, wire.WithHTTPClient(httpClient), wire.WithRequestEditorFn(func(_ context.Context, r *http.Request) error {
		r.Header.Set("Authorization", "Bearer "+apiKey)
		return nil
	}))
	if err != nil {
		return nil, fmt.Errorf("openai: create client: %w", err)
	}
	return &OpenAiGenerator{client: client}, nil
}

var _ Generator = (*OpenAiGenerator)(nil)
var _ StreamingGenerator = (*OpenAiGenerator)(nil)
var _ TokenCounter = (*OpenAiGenerator)(nil)

func convertToolToOpenAI(tool Tool) (wire.ToolDefinition, error) {
	parameters := wire.JSONObject{}
	if tool.InputSchema != nil {
		data, err := json.Marshal(tool.InputSchema)
		if err != nil {
			return wire.ToolDefinition{}, err
		}
		if err := json.Unmarshal(data, &parameters); err != nil {
			return wire.ToolDefinition{}, err
		}
	}
	return wire.ToolDefinition{Type: "function", Function: &wire.FunctionDefinition{Name: &tool.Name, Description: &tool.Description, Parameters: &parameters}}, nil
}

func convertToolsToOpenAI(tools []Tool) ([]wire.ToolDefinition, error) {
	converted := make([]wire.ToolDefinition, 0, len(tools))
	seen := map[string]bool{}
	for _, tool := range tools {
		if tool.Name == "" || tool.Name == "none" || tool.Name == ToolChoiceAuto || tool.Name == ToolChoiceToolsRequired || seen[tool.Name] {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: fmt.Errorf("tool name must be nonempty, unique, and not reserved")}
		}
		seen[tool.Name] = true
		result, err := convertToolToOpenAI(tool)
		if err != nil {
			return nil, &InvalidToolErr{Tool: tool.Name, Cause: err}
		}
		converted = append(converted, result)
	}
	return converted, nil
}

// openAIFields accepts JSON-round-tripped ExtraFields as well as raw maps and
// always returns an owned map, so conversion cannot mutate borrowed messages.
func openAIFields(extra map[string]interface{}) (map[string]json.RawMessage, error) {
	fields := map[string]json.RawMessage{}
	if value, ok := extra[OpenAIExtraFieldWireFields]; ok {
		data, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(data, &fields); err != nil {
			return nil, fmt.Errorf("openai wire fields: %w", err)
		}
	}
	if fields == nil {
		fields = map[string]json.RawMessage{}
	}
	return fields, nil
}

func openAIRawFields(value any, exclude ...string) map[string]json.RawMessage {
	data, _ := json.Marshal(value)
	fields := map[string]json.RawMessage{}
	_ = json.Unmarshal(data, &fields)
	for _, key := range exclude {
		delete(fields, key)
	}
	return fields
}

// openAIPart maps one content block to its wire shape and keeps part-scoped
// provider metadata. Media data and MIME type remain caller-owned inputs.
func openAIPart(block Block) (wire.ContentPart, error) {
	fields, err := openAIFields(block.ExtraFields)
	if err != nil {
		return wire.ContentPart{}, err
	}
	part := wire.ContentPart{AdditionalProperties: fields}
	switch block.ModalityType {
	case Text:
		part.Type = "text"
		text := block.Content.String()
		part.Text = &text
	case Image:
		if block.MimeType == "" {
			return part, fmt.Errorf("image block missing mimetype")
		}
		dataURL := fmt.Sprintf("data:%s;base64,%s", block.MimeType, block.Content.String())
		if block.MimeType == "application/pdf" {
			filename, ok := block.ExtraFields[BlockFieldFilenameKey].(string)
			if !ok {
				return part, fmt.Errorf("filename field missing or not a string")
			}
			part.Type = "file"
			part.File = &wire.FileContent{FileData: &dataURL, Filename: &filename}
		} else {
			part.Type = "image_url"
			var media wire.MediaURL
			data, _ := json.Marshal(map[string]any{"url": dataURL})
			_ = media.UnmarshalJSON(data)
			part.ImageUrl = &media
		}
	case Audio:
		if block.MimeType == "" {
			return part, fmt.Errorf("audio block missing mimetype")
		}
		format, _ := strings.CutPrefix(block.MimeType, "audio/")
		if format != "wav" && format != "mp3" {
			return part, fmt.Errorf("unsupported audio format: %s", block.MimeType)
		}
		data := block.Content.String()
		part.Type = "input_audio"
		part.InputAudio = &wire.InputAudio{Data: &data, Format: &format}
	default:
		return part, UnsupportedInputModalityErr(block.ModalityType.String())
	}
	for _, key := range []string{"type", "text", "image_url", "input_audio", "file"} {
		delete(part.AdditionalProperties, key)
	}
	return part, nil
}

// toOpenAIMessage walks blocks in order, separating tool calls, reasoning, and
// audio references from content. Scalar text is used only when no part metadata
// would be lost. It builds fresh field maps and never mutates borrowed history.
func toOpenAIMessage(msg Message) (wire.Message, error) {
	fields, err := openAIFields(msg.ExtraFields)
	if err != nil {
		return wire.Message{}, err
	}
	result := wire.Message{Role: msg.Role.String(), AdditionalProperties: fields}
	if len(msg.Blocks) == 0 && !(msg.Role == Assistant && len(fields) > 0) {
		return result, fmt.Errorf("message must have at least one block")
	}
	if msg.Role != User && msg.Role != Assistant && msg.Role != ToolResult && msg.Role != System {
		return result, fmt.Errorf("unsupported role: %v", msg.Role)
	}
	if msg.Role == ToolResult {
		result.Role = "tool"
		id := msg.Blocks[0].ID
		if id == "" {
			return result, fmt.Errorf("tool result message block must have an ID")
		}
		result.ToolCallId = &id
	}
	var parts []wire.ContentPart
	var calls []wire.ToolCall
	var reasoning strings.Builder
	readableThinkingSources := make(map[string]bool)
	for _, view := range openAIThinkingViews(fields) {
		readableThinkingSources[view.source] = true
	}
	for _, block := range msg.Blocks {
		switch block.BlockType {
		case Content:
			if msg.Role == ToolResult && block.ID != *result.ToolCallId {
				return result, fmt.Errorf("all blocks in tool result message must have the same ID")
			}
			if msg.Role == Assistant && block.ModalityType == Audio {
				if block.ID == "" {
					return result, fmt.Errorf("assistant audio block missing ID")
				}
				if result.Audio.IsSpecified() {
					return result, fmt.Errorf("multiple assistant audio references cannot be represented")
				}
				result.Audio = nullable.NewNullableWithValue(wire.AudioReference{Id: &block.ID})
				continue
			}
			part, err := openAIPart(block)
			if err != nil {
				return result, err
			}
			parts = append(parts, part)
		case ToolCall:
			if msg.Role != Assistant {
				return result, fmt.Errorf("unsupported block type for %s: %s", msg.Role, block.BlockType)
			}
			var input struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"parameters"`
			}
			if err := json.Unmarshal([]byte(block.Content.String()), &input); err != nil {
				return result, fmt.Errorf("invalid tool call content: %w", err)
			}
			if input.Name == "" || len(input.Arguments) == 0 {
				return result, fmt.Errorf("tool call requires name and arguments")
			}
			args := string(input.Arguments)
			extra, err := openAIFields(block.ExtraFields)
			if err != nil {
				return result, err
			}
			functionFields := map[string]json.RawMessage{}
			if value := extra["function"]; value != nil {
				if err := json.Unmarshal(value, &functionFields); err != nil {
					return result, err
				}
				delete(functionFields, "name")
				delete(functionFields, "arguments")
			}
			for _, key := range []string{"id", "type", "function"} {
				delete(extra, key)
			}
			functionType := "function"
			calls = append(calls, wire.ToolCall{Id: &block.ID, Type: &functionType, Function: &wire.ToolCallFunction{Name: &input.Name, Arguments: &args, AdditionalProperties: functionFields}, AdditionalProperties: extra})
		case Thinking:
			if msg.Role != Assistant {
				return result, fmt.Errorf("reasoning requires assistant role")
			}
			source, _ := block.ExtraFields[OpenAIExtraFieldReasoningSource].(string)
			if !readableThinkingSources[source] {
				reasoning.WriteString(block.Content.String())
			}
		default:
			return result, fmt.Errorf("unsupported block type for %s: %s", msg.Role, block.BlockType)
		}
	}
	for _, key := range []string{"role", "tool_call_id", "tool_calls", "audio"} {
		delete(fields, key)
	}
	if len(parts) > 0 {
		var content wire.MessageContent
		var text strings.Builder
		plainText := true
		for _, part := range parts {
			if part.Type != "text" || part.Text == nil || len(part.AdditionalProperties) != 0 {
				plainText = false
				break
			}
			text.WriteString(*part.Text)
		}
		if plainText {
			_ = content.FromMessageContent0(text.String())
		} else {
			_ = content.FromMessageContent1(parts)
		}
		result.Content = nullable.NewNullableWithValue(content)
		delete(fields, "content")
	}
	if len(calls) > 0 {
		result.ToolCalls = nullable.NewNullableWithValue(calls)
	}
	if reasoning.Len() > 0 {
		text := reasoning.String()
		var existing, alternate string
		_ = json.Unmarshal(fields["reasoning_content"], &existing)
		_ = json.Unmarshal(fields["reasoning"], &alternate)
		if !strings.HasPrefix(existing, text) && !strings.HasPrefix(alternate, text) {
			if existing != "" && !strings.HasPrefix(text, existing) {
				text = existing + text
			}
			result.ReasoningContent = nullable.NewNullableWithValue(text)
			delete(fields, "reasoning_content")
		}
	}
	return result, nil
}

// openAIRequest builds one request from common options and ordered messages.
// Only present options are transmitted; the caller selects the token-limit field
// rather than the adapter inferring provider capabilities from a model name.
func openAIRequest(request GenerationRequest, stream bool) (wire.ChatCompletionRequest, *openAIGenerationOptions, error) {
	params := wire.ChatCompletionRequest{Model: request.Model}
	if request.SafetyIdentifier != "" {
		params.SafetyIdentifier = nullable.NewNullableWithValue(request.SafetyIdentifier)
	}
	if request.PromptCacheKey != "" {
		params.PromptCacheKey = nullable.NewNullableWithValue(request.PromptCacheKey)
	}
	if len(request.Dialog) == 0 {
		return params, nil, ErrEmptyDialog
	}
	options, err := parseOpenAIGenerationOptions(request.Options)
	if err != nil {
		return params, nil, err
	}
	tools, err := convertToolsToOpenAI(request.Tools)
	if err != nil {
		return params, nil, err
	}
	instructions, err := textInstructions(request.Instructions)
	if err != nil {
		return params, nil, err
	}
	if len(instructions) > 0 {
		message, err := toOpenAIMessage(request.Instructions)
		if err != nil {
			return params, nil, err
		}
		// Keep instruction boundaries when text blocks become one string.
		content, contentErr := message.Content.Get()
		if contentErr == nil {
			if _, err := content.AsMessageContent0(); err == nil {
				_ = content.FromMessageContent0(strings.Join(instructions, "\n\n"))
				message.Content = nullable.NewNullableWithValue(content)
			}
		}
		params.Messages = append(params.Messages, message)
	}
	for _, msg := range request.Dialog {
		message, err := toOpenAIMessage(msg)
		if err != nil {
			return params, nil, err
		}
		params.Messages = append(params.Messages, message)
	}
	if len(tools) > 0 {
		params.Tools = nullable.NewNullableWithValue(tools)
	}
	if options.Temperature != nil {
		params.Temperature = nullable.NewNullableWithValue(*options.Temperature)
	}
	if options.TopP != nil {
		params.TopP = nullable.NewNullableWithValue(*options.TopP)
	}
	if options.FrequencyPenalty != nil {
		params.FrequencyPenalty = nullable.NewNullableWithValue(*options.FrequencyPenalty)
	}
	if options.PresencePenalty != nil {
		params.PresencePenalty = nullable.NewNullableWithValue(*options.PresencePenalty)
	}
	field, _, err := generationOption[string](request.Options, OpenAIGenerationOptionTokenLimitField)
	if err != nil {
		return params, nil, err
	}
	if field != "" && field != "max_tokens" && field != "max_completion_tokens" {
		return params, nil, InvalidParameterErr{Parameter: OpenAIGenerationOptionTokenLimitField, Reason: "expected max_tokens or max_completion_tokens"}
	}
	if options.MaxGenerationTokens != nil {
		value := nullable.NewNullableWithValue(int64(*options.MaxGenerationTokens))
		if field == "max_tokens" {
			params.MaxTokens = value
		} else {
			params.MaxCompletionTokens = value
		}
	}
	if options.CandidateCount != nil {
		params.N = nullable.NewNullableWithValue(int64(*options.CandidateCount))
	}
	if len(options.StopSequences) > 0 {
		var stop wire.StopSequences
		if len(options.StopSequences) == 1 {
			_ = stop.FromStopSequences0(options.StopSequences[0])
		} else {
			_ = stop.FromStopSequences1(options.StopSequences)
		}
		params.Stop = nullable.NewNullableWithValue(stop)
	}
	if options.ToolChoice != "" {
		var choice wire.ToolChoice
		if options.ToolChoice == "none" || options.ToolChoice == ToolChoiceAuto || options.ToolChoice == ToolChoiceToolsRequired {
			_ = choice.FromToolChoice0(options.ToolChoice)
		} else {
			functionType := "function"
			_ = choice.FromToolChoiceObject(wire.ToolChoiceObject{Type: &functionType, Function: &wire.NamedTool{Name: &options.ToolChoice}})
		}
		params.ToolChoice = nullable.NewNullableWithValue(choice)
	}
	if options.ReasoningEffort != "" {
		params.ReasoningEffort = nullable.NewNullableWithValue(options.ReasoningEffort)
	}
	var modalities []string
	for _, modality := range options.OutputModalities {
		if modality != Text && modality != Audio {
			return params, nil, UnsupportedOutputModalityErr(modality.String() + " output not supported by model")
		}
		modalities = append(modalities, modality.String())
		if modality == Audio {
			if options.AudioConfig.VoiceName == "" {
				return params, nil, InvalidParameterErr{Parameter: "OpenAIAudioConfig.VoiceName", Reason: "voice name is required for audio output"}
			}
			if options.AudioConfig.Format == "" {
				return params, nil, InvalidParameterErr{Parameter: "OpenAIAudioConfig.Format", Reason: "format is required for audio output"}
			}
			params.Audio = nullable.NewNullableWithValue(wire.AudioOptions{Voice: &options.AudioConfig.VoiceName, Format: &options.AudioConfig.Format})
		}
	}
	if len(modalities) > 0 {
		params.Modalities = nullable.NewNullableWithValue(modalities)
	}
	include, specified, err := generationOption[bool](request.Options, OpenAIGenerationOptionStreamUsage)
	if err != nil {
		return params, nil, err
	}
	if stream {
		params.Stream = nullable.NewNullableWithValue(true)
		if !specified || include {
			include = true
			params.StreamOptions = nullable.NewNullableWithValue(wire.StreamOptions{IncludeUsage: &include})
		}
	}
	fields, _, err := generationOption[map[string]json.RawMessage](request.Options, OpenAIGenerationOptionExtraBody)
	if err != nil {
		return params, nil, err
	}
	params.AdditionalProperties = make(map[string]json.RawMessage, len(fields))
	if options.LogitBias != nil {
		value, err := json.Marshal(*options.LogitBias)
		if err != nil {
			return params, nil, &InvalidParameterErr{Parameter: GenerationOptionLogitBias, Reason: err.Error()}
		}
		params.AdditionalProperties["logit_bias"] = value
	}
	if options.Logprobs != nil {
		params.Logprobs = nullable.NewNullableWithValue(*options.Logprobs)
	}
	if options.TopLogprobs != nil {
		params.TopLogprobs = nullable.NewNullableWithValue(int64(*options.TopLogprobs))
	}
	if options.ParallelToolCalls != nil {
		params.ParallelToolCalls = nullable.NewNullableWithValue(*options.ParallelToolCalls)
	}
	if options.Seed != nil {
		params.Seed = nullable.NewNullableWithValue(int64(*options.Seed))
	}
	for name, value := range fields {
		switch name {
		case "model", "messages", "tools", "stream":
			return params, nil, InvalidParameterErr{Parameter: OpenAIGenerationOptionExtraBody, Reason: "cannot override " + name}
		}
		if !json.Valid(value) {
			return params, nil, InvalidParameterErr{Parameter: OpenAIGenerationOptionExtraBody, Reason: "invalid JSON for " + name}
		}
		params.AdditionalProperties[name] = bytes.Clone(value)
	}
	return params, options, nil
}

func openAIUsage(usage wire.Usage) Metadata {
	result := Metadata{}
	if usage.PromptTokens != nil {
		result[UsageMetricInputTokens] = int(*usage.PromptTokens)
	}
	if usage.CompletionTokens != nil {
		result[UsageMetricGenerationTokens] = int(*usage.CompletionTokens)
	}
	details, _ := usage.PromptTokensDetails.Get()
	cached := details.CachedTokens
	if cached == nil {
		cached = usage.CachedTokens
	}
	if cached == nil {
		cached = usage.PromptCacheHitTokens
	}
	if cached != nil {
		result[UsageMetricCacheReadTokens] = int(*cached)
	}
	if details.CacheWriteTokens != nil {
		result[UsageMetricCacheWriteTokens] = int(*details.CacheWriteTokens)
	}
	completion, _ := usage.CompletionTokensDetails.Get()
	if completion.ReasoningTokens != nil {
		result[UsageMetricReasoningTokens] = int(*completion.ReasoningTokens)
	}
	return result
}

type openAIThinkingView struct{ source, text string }

// openAIThinkingDetails extracts text and string summaries from reasoning_details.
// Each result has the same position as its input item. Encrypted data, unknown
// types, and items without a type produce an empty string, not display text.
func openAIThinkingDetails(fields map[string]json.RawMessage) []string {
	var details []wire.ReasoningDetail
	if json.Unmarshal(fields["reasoning_details"], &details) != nil {
		return nil
	}
	texts := make([]string, len(details))
	for i, detail := range details {
		if detail.Type == nil {
			continue
		}
		switch *detail.Type {
		case "reasoning.text":
			texts[i], _ = detail.Text.Get()
		case "reasoning.summary":
			if raw, err := detail.Summary.Get(); err == nil {
				_ = json.Unmarshal(raw, &texts[i])
			}
		}
	}
	return texts
}

// openAIThinkingViews collects the readable text from each reasoning field.
// A provider can include the same text in several fields; this function leaves
// duplicate removal to the display code and does not change the original data.
func openAIThinkingViews(fields map[string]json.RawMessage) []openAIThinkingView {
	var views []openAIThinkingView
	if text := strings.Join(openAIThinkingDetails(fields), ""); text != "" {
		views = append(views, openAIThinkingView{"reasoning_details", text})
	}
	for _, source := range []string{"reasoning_content", "reasoning"} {
		var text string
		if json.Unmarshal(fields[source], &text) == nil && text != "" {
			views = append(views, openAIThinkingView{source, text})
		}
	}
	return views
}

// openAIThinkingDisplay tracks reasoning text already sent to the caller for
// one candidate. It avoids repeating text returned in several fields without
// dropping additional text that arrives later. It never changes the original
// reasoning data kept for the next request.
type openAIThinkingDisplay struct {
	views []string
}

// update returns thinking blocks for newly received text. For example, after
// emitting "First ", receiving "First step" adds only "step". If another field
// instead contains "A different approach", that text gets its own block.
func (d *openAIThinkingDisplay) update(fields map[string]json.RawMessage) []Block {
	var blocks []Block
	for _, view := range openAIThinkingViews(fields) {
		if text := d.unseen(view.text); text != "" {
			blocks = append(blocks, Block{BlockType: Thinking, ModalityType: Text, MimeType: "text/plain", Content: Str(text), ExtraFields: map[string]interface{}{OpenAIExtraFieldReasoningSource: view.source}})
		}
	}
	return blocks
}

// unseen skips repeated text and returns only the added part when a field grows.
// Different text is returned in full; a stream cannot replace earlier output.
func (d *openAIThinkingDisplay) unseen(text string) string {
	longest := 0
	for _, previous := range d.views {
		if strings.HasPrefix(previous, text) {
			return ""
		}
		if strings.HasPrefix(text, previous) {
			longest = max(longest, len(previous))
		}
	}
	kept := d.views[:0]
	for _, previous := range d.views {
		if !strings.HasPrefix(text, previous) {
			kept = append(kept, previous)
		}
	}
	d.views = append(kept, text)
	return text[longest:]
}

// openAIResponseMessage normalizes text, audio, reasoning, and function calls
// without decoding arbitrary argument numbers through float64. Remaining wire
// fields stay at their original message or block scope for explicit replay.
func openAIResponseMessage(message wire.ResponseMessage, audioFormat string) (Message, error) {
	result := Message{Role: Assistant}
	fields := openAIRawFields(message, "role", "tool_calls", "audio")
	if message.Content.IsSpecified() && !message.Content.IsNull() {
		content, _ := message.Content.Get()
		if text, err := content.AsMessageContent0(); err == nil {
			if text != "" {
				result.Blocks = append(result.Blocks, TextBlock(text))
				delete(fields, "content")
			}
		} else {
			parts, err := content.AsMessageContent1()
			if err != nil {
				return result, fmt.Errorf("decode content parts: %w", err)
			}
			for _, part := range parts {
				if part.Type != "text" || part.Text == nil {
					return result, fmt.Errorf("unsupported output content part: %s", part.Type)
				}
				block := TextBlock(*part.Text)
				extra := openAIRawFields(part, "type", "text")
				if len(extra) > 0 {
					block.ExtraFields = map[string]interface{}{OpenAIExtraFieldWireFields: extra}
				}
				result.Blocks = append(result.Blocks, block)
			}
			delete(fields, "content")
		}
	}
	var thinking openAIThinkingDisplay
	result.Blocks = append(result.Blocks, thinking.update(fields)...)
	if audio, err := message.Audio.Get(); err == nil && audio.Id != nil && *audio.Id != "" {
		block := Block{ID: *audio.Id, BlockType: Content, ModalityType: Audio, MimeType: "audio/" + audioFormat,
			ExtraFields: map[string]interface{}{OpenAIExtraFieldAudio: openAIRawFields(audio, "id", "data")}}
		if audio.Data != nil {
			block.Content = Str(*audio.Data)
		}
		result.Blocks = append(result.Blocks, block)
		if audio.Transcript != nil && *audio.Transcript != "" {
			result.Blocks = append(result.Blocks, TextBlock(*audio.Transcript))
		}
	}
	if len(fields) > 0 {
		result.ExtraFields = map[string]interface{}{OpenAIExtraFieldWireFields: fields}
	}
	calls, _ := message.ToolCalls.Get()
	for _, call := range calls {
		if call.Function == nil || call.Function.Name == nil || call.Function.Arguments == nil {
			return result, fmt.Errorf("unsupported or incomplete tool call")
		}
		args := *call.Function.Arguments
		var parameters map[string]json.RawMessage
		if err := json.Unmarshal([]byte(args), &parameters); err != nil {
			return result, fmt.Errorf("failed to parse tool arguments: %w", err)
		}
		payload, err := json.Marshal(struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"parameters"`
		}{*call.Function.Name, json.RawMessage(args)})
		if err != nil {
			return result, err
		}
		block := Block{BlockType: ToolCall, ModalityType: Text, MimeType: "application/json", Content: Str(payload)}
		if call.Id != nil {
			block.ID = *call.Id
		}
		extra := openAIRawFields(call, "id", "type", "function")
		if functionFields := openAIRawFields(call.Function, "name", "arguments"); len(functionFields) > 0 {
			extra["function"], _ = json.Marshal(functionFields)
		}
		block.ExtraFields = map[string]interface{}{OpenAIExtraFieldWireFields: extra}
		result.Blocks = append(result.Blocks, block)
	}
	return result, nil
}

func openAIFinish(reason, refusal string, tools bool) (FinishReason, error) {
	if refusal != "" {
		return ContentPolicyViolation, ContentPolicyErr(refusal)
	}
	switch reason {
	case "stop":
		if tools {
			return ToolUse, nil
		}
		return EndTurn, nil
	case "tool_calls":
		return ToolUse, nil
	case "length", "model_context_window_exceeded":
		return MaxGenerationLimit, ErrMaxGenerationLimit
	case "content_filter", "sensitive":
		return ContentPolicyViolation, ContentPolicyErr("content policy violation detected")
	case "error":
		return Unknown, &ApiErr{Provider: ProviderOpenAI, Kind: APIErrorKindServer, Message: "generation stopped: error"}
	case "network_error", "insufficient_system_resource":
		// Compatible providers use these terminal reasons instead of an HTTP
		// error. Wrappers replace Provider while preserving retry classification.
		return Unknown, &ApiErr{Provider: ProviderOpenAI, Kind: APIErrorKindServiceUnavailable, Message: "generation stopped: " + reason}
	default:
		return Unknown, nil
	}
}

// openAIResponseError reads an error delivered after HTTP headers were sent.
// Some compatible APIs put it on a choice beside partial output. Keep the full
// response body for inspection, but use that error's message and numeric status.
func openAIResponseError(response *http.Response, data []byte) *ApiErr {
	var envelope struct {
		Error   json.RawMessage `json:"error"`
		Choices []struct {
			Error json.RawMessage `json:"error"`
		} `json:"choices"`
	}
	_ = json.Unmarshal(data, &envelope)
	detail := envelope.Error
	if len(detail) == 0 || bytes.Equal(bytes.TrimSpace(detail), []byte("null")) {
		for _, choice := range envelope.Choices {
			if len(choice.Error) > 0 && !bytes.Equal(bytes.TrimSpace(choice.Error), []byte("null")) {
				detail = choice.Error
				break
			}
		}
	}
	if len(detail) == 0 || bytes.Equal(bytes.TrimSpace(detail), []byte("null")) {
		detail = data
	}
	var fields struct {
		Code int `json:"code"`
	}
	_ = json.Unmarshal(detail, &fields)
	status := response.StatusCode
	if status >= 200 && status < 300 && fields.Code >= 400 && fields.Code <= 599 {
		status = fields.Code
	}
	return &ApiErr{
		Provider:           ProviderOpenAI,
		Kind:               classifyHTTPStatus(status),
		StatusCode:         status,
		Message:            parseAPIErrorMessage(string(detail)),
		RawBody:            string(data),
		RetryAfterDuration: retryAfterFromResponse(response),
	}
}

// Generate sends one request, preserving response headers and opaque metadata.
// If the provider reports an error alongside partial output, both the response
// and the error are returned. Check err before treating the answer as complete.
func (g *OpenAiGenerator) Generate(ctx context.Context, request GenerationRequest) (Response, error) {
	if g.client == nil {
		return Response{}, fmt.Errorf("openai: client not initialized")
	}
	params, options, err := openAIRequest(request, false)
	if err != nil {
		return Response{}, err
	}
	response, err := g.client.CreateChatCompletion(ctx, nil, params, requestHeaderEditor(request.Headers))
	if err != nil {
		return Response{}, fmt.Errorf("openai request: %w", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return Response{}, mapHTTPAPIError(ProviderOpenAI, response)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		return Response{}, fmt.Errorf("openai response: %w", err)
	}
	var completion wire.ChatCompletionResponse
	if err := json.Unmarshal(data, &completion); err != nil {
		return Response{}, fmt.Errorf("decode openai response: %w", err)
	}
	var generationErr error
	if completion.Error.IsSpecified() && !completion.Error.IsNull() {
		generationErr = openAIResponseError(response, data)
	}
	usage, _ := completion.Usage.Get()
	result := Response{
		UsageMetadata: openAIUsage(usage),
		ExtraFields: map[string]interface{}{
			OpenAIResponseExtraFieldHeaders:    response.Header.Clone(),
			OpenAIResponseExtraFieldUsage:      openAIRawFields(usage),
			OpenAIResponseExtraFieldWireFields: openAIRawFields(completion, "choices", "usage"),
		},
	}
	if completion.Choices == nil {
		if generationErr != nil {
			return result, generationErr
		}
		return result, fmt.Errorf("openai response missing choices")
	}
	for _, choice := range *completion.Choices {
		detail := choice.AdditionalProperties["error"]
		if len(detail) > 0 && !bytes.Equal(bytes.TrimSpace(detail), []byte("null")) {
			generationErr = openAIResponseError(response, data)
		}
		if choice.Message == nil {
			if generationErr != nil {
				continue
			}
			return result, fmt.Errorf("openai choice missing message")
		}
		message, err := openAIResponseMessage(*choice.Message, options.AudioConfig.Format)
		if err != nil && generationErr == nil {
			return result, err
		}
		// A provider failure can interrupt tool arguments. Keep the text and
		// metadata decoded so far, but return the provider error, not a JSON
		// parsing error. The incomplete tool call is never added to the message.
		if message.ExtraFields == nil {
			message.ExtraFields = map[string]interface{}{}
		}
		message.ExtraFields[OpenAIExtraFieldChoice] = openAIRawFields(choice, "message")
		result.Candidates = append(result.Candidates, message)
	}
	if generationErr != nil {
		return result, generationErr
	}
	if len(*completion.Choices) > 0 {
		first := (*completion.Choices)[0]
		reason, _ := first.FinishReason.Get()
		refusal, _ := first.Message.Refusal.Get()
		calls, _ := first.Message.ToolCalls.Get()
		result.FinishReason, err = openAIFinish(reason, refusal, len(calls) > 0)
	}
	return result, err
}

type openAIGenerationOptions struct {
	LogitBias           *map[string]float64
	Logprobs            *bool
	TopLogprobs         *int
	ParallelToolCalls   *bool
	Seed                *int
	Temperature         *float64
	TopP                *float64
	FrequencyPenalty    *float64
	PresencePenalty     *float64
	CandidateCount      *uint
	MaxGenerationTokens *int
	ToolChoice          string
	StopSequences       []string
	OutputModalities    []Modality
	AudioConfig         OpenAIAudioConfig
	ReasoningEffort     string
}

// parseOpenAIGenerationOptions validates common option values and records the typed Chat Completions configuration.
func parseOpenAIGenerationOptions(values GenerationOptions) (*openAIGenerationOptions, error) {
	options := &openAIGenerationOptions{}
	logitBias, present, err := generationOption[map[string]float64](values, GenerationOptionLogitBias)
	if err != nil {
		return nil, err
	}
	if present {
		options.LogitBias = &logitBias
	}
	logprobs, present, err := generationOption[bool](values, GenerationOptionLogprobs)
	if err != nil {
		return nil, err
	}
	if present {
		options.Logprobs = &logprobs
	}
	topLogprobs, present, err := generationOption[int](values, GenerationOptionTopLogprobs)
	if err != nil {
		return nil, err
	}
	if present {
		options.TopLogprobs = &topLogprobs
	}
	parallelToolCalls, present, err := generationOption[bool](values, GenerationOptionParallelToolCalls)
	if err != nil {
		return nil, err
	}
	if present {
		options.ParallelToolCalls = &parallelToolCalls
	}
	seed, present, err := generationOption[int](values, GenerationOptionSeed)
	if err != nil {
		return nil, err
	}
	if present {
		options.Seed = &seed
	}

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
	frequencyPenalty, ok, err := generationOption[float64](values, GenerationOptionFrequencyPenalty)
	if err != nil {
		return nil, err
	}
	if ok {
		options.FrequencyPenalty = &frequencyPenalty
	}
	presencePenalty, ok, err := generationOption[float64](values, GenerationOptionPresencePenalty)
	if err != nil {
		return nil, err
	}
	if ok {
		options.PresencePenalty = &presencePenalty
	}
	candidateCount, ok, err := generationOption[uint](values, GenerationOptionCandidateCount)
	if err != nil {
		return nil, err
	}
	if ok {
		options.CandidateCount = &candidateCount
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
	if options.OutputModalities, _, err = generationOption[[]Modality](values, OpenAIGenerationOptionOutputModalities); err != nil {
		return nil, err
	}
	if options.AudioConfig, _, err = generationOption[OpenAIAudioConfig](values, OpenAIGenerationOptionAudioConfig); err != nil {
		return nil, err
	}
	if options.ReasoningEffort, _, err = generationOption[string](values, GenerationOptionReasoningEffort); err != nil {
		return nil, err
	}
	return options, nil
}

// calculateImageTokens calculates the number of tokens used by an image block
// based on OpenAI's token calculation rules for different models.
//
// The function first attempts to extract dimensions directly from the image data by:
//  1. Decoding the base64-encoded image content
//  2. Using Go's image package to determine width and height
//
// If extraction from image data fails, it tries to get dimensions from ExtraFields.
// If dimensions still cannot be determined, an error is returned.
//
// PDFs are not supported for token counting as they are converted to images server-side
// and exact dimensions cannot be determined.
//
// The detail level ("high" or "low") is determined from ExtraFields if specified,
// with "high" being the default.
//
// Token calculation depends on the model:
//   - For minimal models (GPT-4.1-mini, GPT-4.1-nano, o4-mini), it uses the 32px patch method
//   - For standard models (GPT-4o, etc.), it uses the base+tile method with detail level consideration
//
// Returns:
//   - The number of tokens as an integer
//   - An error if dimensions cannot be determined or if calculation fails
func (g *OpenAiGenerator) calculateImageTokens(block Block, model string) (int, error) {
	if block.MimeType == "application/pdf" {
		return 0, fmt.Errorf("PDF token counting is not supported")
	}

	var width, height int
	var detail string = "high" // Default to high detail

	// Try to extract image dimensions directly from the image data
	imgData, err := base64.StdEncoding.DecodeString(block.Content.String())
	if err == nil {
		// Successfully decoded base64, now try to extract dimensions
		imgReader := bytes.NewReader(imgData)
		config, _, err := image.DecodeConfig(imgReader)
		if err == nil {
			// Successfully extracted dimensions
			width = config.Width
			height = config.Height
		}
	}

	// If we couldn't extract dimensions from the image, try the ExtraFields
	if width == 0 || height == 0 {
		if block.ExtraFields != nil {
			if w, ok := block.ExtraFields[OpenAIExtraFieldImageWidth].(int); ok {
				width = w
			}
			if h, ok := block.ExtraFields[OpenAIExtraFieldImageHeight].(int); ok {
				height = h
			}
		}
	}

	// Return an error if we still couldn't determine dimensions
	if width == 0 || height == 0 {
		return 0, fmt.Errorf("could not determine image dimensions for token calculation")
	}

	// Get detail level from ExtraFields if specified
	if block.ExtraFields != nil {
		if d, ok := block.ExtraFields[OpenAIExtraFieldImageDetail].(string); ok && (d == "low" || d == "high") {
			detail = d
		}
	}

	// Determine which calculation method to use based on model
	if isMinimalModel(model) {
		return calculateMinimalModelImageTokens(width, height, model)
	} else {
		return calculateStandardModelImageTokens(width, height, detail, model)
	}
}

// isMinimalModel checks if the model uses the 32px patch calculation method.
// It returns true for models that use patch-based calculation like gpt-4.1-mini,
// gpt-4.1-nano, and o4-mini.
//
// These models use a different token calculation algorithm that divides images
// into 32x32 pixel patches and applies model-specific multipliers.
func isMinimalModel(model string) bool {
	minimalModels := []string{"gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"}
	for _, m := range minimalModels {
		if strings.Contains(model, m) {
			return true
		}
	}
	return false
}

// calculateMinimalModelImageTokens calculates image tokens for minimal models
// (GPT-4.1-mini, GPT-4.1-nano, o4-mini) based on OpenAI's token calculation rules.
//
// The calculation follows these steps:
//  1. Calculate the number of 32x32 pixel patches needed to cover the image
//  2. If the number exceeds 1536, scale the image to fit within that limit
//  3. Apply a model-specific multiplier to the patch count
//
// Multipliers:
//   - GPT-4.1-mini: 1.62
//   - GPT-4.1-nano: 2.46
//   - o4-mini:      1.72
//
// The result is the final token count for the image.
func calculateMinimalModelImageTokens(width, height int, model string) (int, error) {
	// Calculate patches based on 32px x 32px
	patchesWidth := (width + 32 - 1) / 32   // Ceiling division
	patchesHeight := (height + 32 - 1) / 32 // Ceiling division

	totalPatches := patchesWidth * patchesHeight

	// If exceeds 1536, scale down
	if totalPatches > 1536 {
		// We need to scale down the image while preserving aspect ratio
		// Calculate shrink factor
		shrinkFactor := math.Sqrt(float64(1536*32*32) / float64(width*height))

		// Apply shrink factor
		scaledWidth := int(float64(width) * shrinkFactor)
		scaledHeight := int(float64(height) * shrinkFactor)

		if scaledWidth%32 != 0 {
			widthPatches := float64(scaledWidth) / 32.0
			shrinkFactor = math.Floor(widthPatches) / widthPatches

			// Apply shrink factor with dimensions that fit in a whole patch size (for width)
			scaledWidth = int(float64(scaledWidth) * shrinkFactor)
			scaledHeight = int(float64(scaledHeight) * shrinkFactor)
		}

		// Recalculate patches
		patchesWidth = (scaledWidth + 32 - 1) / 32
		patchesHeight = (scaledHeight + 32 - 1) / 32

		totalPatches = patchesWidth * patchesHeight
	}

	// Apply multiplier based on model
	switch {
	case strings.Contains(model, "gpt-4.1-mini"):
		return int(float64(totalPatches) * 1.62), nil
	case strings.Contains(model, "gpt-4.1-nano"):
		return int(float64(totalPatches) * 2.46), nil
	case strings.Contains(model, "o4-mini"):
		return int(float64(totalPatches) * 1.72), nil
	default:
		// Default to no multiplier if model is in minimal category but not recognized
		return totalPatches, nil
	}
}

// calculateStandardModelImageTokens calculates image tokens for standard models
// (GPT-4o, GPT-4.1, etc.) based on OpenAI's token calculation rules.
//
// For "low" detail images, it returns a fixed base token count dependent on the model.
//
// For "high" detail images, the calculation follows these steps:
//  1. Scale the image to fit within a 2048x2048 square if necessary
//  2. Scale so the shortest side is 768px
//  3. Count the number of 512px tiles needed to cover the image
//  4. Calculate tokens as: base_tokens + (tile_tokens * number_of_tiles)
//
// Base and tile token counts vary by model and are retrieved using getTokensForModel().
//
// Returns the calculated token count for the image.
func calculateStandardModelImageTokens(width, height int, detail string, model string) (int, error) {
	// For low detail, return base tokens
	if detail == "low" {
		baseTokens, _ := getTokensForModel(model)
		return baseTokens, nil
	}

	// For high detail, we need to calculate based on tiles
	// First, get base and tile tokens for the model
	baseTokens, tileTokens := getTokensForModel(model)

	// Scale to fit in 2048px x 2048px square if needed
	scaledWidth, scaledHeight := width, height
	if width > 2048 || height > 2048 {
		// Scale down preserving aspect ratio
		ratio := float64(2048) / math.Max(float64(width), float64(height))
		scaledWidth = int(float64(width) * ratio)
		scaledHeight = int(float64(height) * ratio)
	}

	// Scale so shortest side is 768px
	ratio := float64(768) / math.Min(float64(scaledWidth), float64(scaledHeight))
	scaledWidth = int(float64(scaledWidth) * ratio)
	scaledHeight = int(float64(scaledHeight) * ratio)

	// Count 512px tiles
	tilesWidth := (scaledWidth + 512 - 1) / 512   // Ceiling division
	tilesHeight := (scaledHeight + 512 - 1) / 512 // Ceiling division

	totalTiles := tilesWidth * tilesHeight

	// Calculate total tokens
	return baseTokens + (tileTokens * totalTiles), nil
}

// getTokensForModel returns the base tokens and tile tokens for a given model.
// These values are used in token calculations for standard models (non-minimal models)
// according to OpenAI's documentation.
//
// Model-specific token values:
//   - GPT-4o, GPT-4.1, GPT-4.5:        base=85,  tile=170
//   - GPT-4o-mini:                      base=2833, tile=5667
//   - o1, o1-pro, o3:                   base=75,  tile=150
//   - computer-use-preview:             base=65,  tile=129
//   - Default (unrecognized models):    base=85,  tile=170 (Same as GPT-4o)
//
// Returns the base tokens and tile tokens as integers.
func getTokensForModel(model string) (baseTokens, tileTokens int) {
	switch {
	case strings.Contains(model, "4o") ||
		strings.Contains(model, "4.1") ||
		strings.Contains(model, "4.5"):
		return 85, 170
	case strings.Contains(model, "4o-mini"):
		return 2833, 5667
	case strings.Contains(model, "o1") ||
		strings.Contains(model, "o1-pro") ||
		strings.Contains(model, "o3"):
		return 75, 150
	case strings.Contains(model, "computer-use-preview"):
		return 65, 129
	default:
		// Default to GPT-4o values
		return 85, 170
	}
}

// Count uses tiktoken to count the request model, instructions, dialog, and
// tools without a provider call.
//
// Image counts use dimensions decoded from image data or supplied through
// [OpenAIExtraFieldImageWidth] and [OpenAIExtraFieldImageHeight]. [PDFBlock]
// counting is unsupported because server-side page conversion is not observable.
func (g *OpenAiGenerator) Count(ctx context.Context, request GenerationRequest) (uint, error) {
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	default:
	}

	dialog := request.Dialog
	if len(dialog) == 0 {
		return 0, ErrEmptyDialog
	}

	tke, err := tiktoken.EncodingForModel(request.Model)
	if err != nil {
		tke, err = tiktoken.GetEncoding(tiktoken.MODEL_O200K_BASE) // Fallback
		if err != nil {
			return 0, fmt.Errorf("failed to get tiktoken encoding: %w", err)
		}
	}

	var totalTokens int

	instructions, err := textInstructions(request.Instructions)
	if err != nil {
		return 0, err
	}
	for _, instruction := range instructions {
		totalTokens += len(tke.Encode(instruction, nil, nil))
	}

	// See https://platform.openai.com/docs/guides/images-vision?api-mode=chat#calculating-costs
	for _, msg := range dialog {
		for _, block := range msg.Blocks {
			switch block.ModalityType {
			case Text:
				contentToTokenize := block.Content.String()
				totalTokens += len(tke.Encode(contentToTokenize, nil, nil))
			case Image:
				// Extract dimensions from the image content
				imageTokens, err := g.calculateImageTokens(block, request.Model)
				if err != nil {
					return 0, fmt.Errorf("failed to calculate image tokens: %w", err)
				}
				totalTokens += imageTokens
			default:
				return 0, UnsupportedOutputModalityErr("unsupported modality")
			}
		}
	}

	tools, err := convertToolsToOpenAI(request.Tools)
	if err != nil {
		return 0, err
	}
	for _, tool := range tools {
		var toolDefStr string
		toolDefStr += *tool.Function.Name + "\n"
		toolDefStr += *tool.Function.Description + "\n"
		paramsJSON, err := json.Marshal(tool.Function.Parameters)
		if err == nil {
			toolDefStr += string(paramsJSON) + "\n"
		}
		totalTokens += len(tke.Encode(toolDefStr, nil, nil))
	}

	return uint(totalTokens), nil
}
