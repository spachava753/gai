package gai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
)

// Role identifies how a [Message] participates in a [GenerationRequest]. The
// zero value is [User].
type Role uint

const (
	// User identifies caller-authored conversational input in [GenerationRequest.Dialog].
	User Role = iota

	// Assistant identifies model output. Assistant messages can be returned in
	// [Response.Candidates] or replayed in a later [Dialog].
	Assistant

	// ToolResult identifies application-owned output for a preceding [ToolCall].
	// Construct tool results with [ToolResultMessage].
	ToolResult

	// System identifies model instructions. System messages belong in
	// [GenerationRequest.Instructions], outside [GenerationRequest.Dialog].
	System
)

// String returns the wire-oriented name of r. Unknown values include their
// numeric value in the returned string.
func (r Role) String() string {
	switch r {
	case User:
		return "user"
	case Assistant:
		return "assistant"
	case ToolResult:
		return "tool result"
	case System:
		return "system"
	default:
		return fmt.Sprintf("unknown role %d", r)
	}
}

// Modality identifies the media represented by a [Block]. The zero value is
// [Text]. Provider types document which input and output modalities they accept.
type Modality uint

const (
	// Text identifies textual content, reasoning, metadata, and tool payloads.
	Text Modality = iota
	// Image identifies image input or output.
	Image
	// Audio identifies audio input or output.
	Audio
	// Video identifies video input or output.
	Video
)

// String returns the lowercase name of m. Unknown values include their numeric
// value in the returned string.
func (m Modality) String() string {
	switch m {
	case Text:
		return "text"
	case Image:
		return "image"
	case Audio:
		return "audio"
	case Video:
		return "video"
	default:
		return fmt.Sprintf("unknown modality %d", m)
	}
}

const (
	// Content is the [Block.BlockType] for ordinary text or media content.
	Content = "content"

	// Thinking is the [Block.BlockType] for model reasoning. Thinking blocks set
	// [ThinkingExtraFieldGeneratorKey] and can carry provider replay metadata in
	// [Block.ExtraFields].
	Thinking = "thinking"

	// ToolCall is the [Block.BlockType] for a model-requested function call. Its
	// content encodes [ToolCallInput].
	ToolCall = "tool_call"

	// MetadataBlockType is the [Block.BlockType] used by streams to emit
	// [Metadata] before successful termination.
	MetadataBlockType = "metadata"

	// Separator is the internal [Block.BlockType] used to preserve logical stream
	// boundaries. [StreamingAdapter] consumes and removes separator blocks.
	Separator = "separator"

	// ThinkingExtraFieldGeneratorKey is the [Block.ExtraFields] key that identifies
	// the provider adapter that produced a [Thinking] block. Its string value is
	// one of the ThinkingGenerator constants below.
	ThinkingExtraFieldGeneratorKey = "thinking_generator"

	// ThinkingGeneratorAnthropic is the [ThinkingExtraFieldGeneratorKey] value for
	// [AnthropicGenerator]. See [AnthropicExtraFieldThinkingSignature].
	ThinkingGeneratorAnthropic = "anthropic"

	// ThinkingGeneratorCerebras is the [ThinkingExtraFieldGeneratorKey] value for
	// [CerebrasGenerator].
	ThinkingGeneratorCerebras = "cerebras"

	// ThinkingGeneratorDeepSeek is the [ThinkingExtraFieldGeneratorKey] value for
	// [DeepSeekGenerator].
	ThinkingGeneratorDeepSeek = "deepseek"

	// ThinkingGeneratorGemini is the [ThinkingExtraFieldGeneratorKey] value for
	// [GeminiGenerator]. See [GeminiExtraFieldThoughtSignature].
	ThinkingGeneratorGemini = "gemini"

	// ThinkingGeneratorOpenRouter is the [ThinkingExtraFieldGeneratorKey] value
	// for [OpenRouterGenerator]. See [OpenRouterExtraFieldReasoningType].
	ThinkingGeneratorOpenRouter = "openrouter"

	// ThinkingGeneratorResponses is the [ThinkingExtraFieldGeneratorKey] value
	// for [ResponsesGenerator].
	ThinkingGeneratorResponses = "responses"

	// ThinkingGeneratorZai is the [ThinkingExtraFieldGeneratorKey] value for
	// [ZaiGenerator].
	ThinkingGeneratorZai = "zai"
)

// Block is one ordered unit of message content. A block can contain ordinary
// [Content], model [Thinking], a [ToolCall], stream [MetadataBlockType], or an
// internal [Separator]. Use the block constructors for common input forms.
type Block struct {
	// ID correlates tool calls and tool results and can carry a provider content
	// identifier. An empty string means no identifier is attached.
	ID string `json:"id,omitempty" yaml:"id,omitempty"`

	// BlockType identifies the block contract. Set it to [Content], [Thinking],
	// [ToolCall], [MetadataBlockType], or [Separator]. The empty string is not a
	// valid input block type.
	BlockType string `json:"block_type" yaml:"block_type"`

	// ModalityType identifies the media in Content. The zero value is [Text].
	ModalityType Modality `json:"modality_type" yaml:"modality_type"`

	// MimeType identifies the content encoding, such as "text/plain",
	// "image/png", or "audio/mpeg". Block constructors set an appropriate value;
	// provider adapters can reject missing or unsupported types.
	MimeType string `json:"mime_type,omitempty" yaml:"mime_type,omitempty"`

	// Content is the block payload. Text content is normally [Str]. Binary input
	// helpers base64-encode raw bytes and store the encoded value as Str. A nil
	// value is invalid for request content.
	Content fmt.Stringer `json:"content,omitempty" yaml:"content,omitempty"`

	// ExtraFields carries provider or block-specific data at the narrowest replay
	// scope. Examples include [ThinkingExtraFieldGeneratorKey],
	// [AnthropicExtraFieldThinkingSignature], [GeminiExtraFieldThoughtSignature],
	// [ResponsesExtraFieldReasoningID], and [BlockFieldFilenameKey].
	ExtraFields map[string]interface{} `json:"extra_fields,omitempty" yaml:"extra_fields,omitempty"`
}

// Message is one role-scoped, ordered sequence of [Block] values. Messages in
// [GenerationRequest.Dialog] use [User], [Assistant], or [ToolResult]. A
// [System] message belongs in [GenerationRequest.Instructions].
type Message struct {
	// Role determines how providers interpret Blocks. The zero value is [User],
	// but callers should set it explicitly.
	Role Role `json:"role" yaml:"role"`

	// Blocks preserves the provider-visible content order.
	Blocks []Block `json:"blocks" yaml:"blocks"`

	// ToolResultError marks a [ToolResult] message as an unsuccessful tool
	// execution whose content should be shown to the model.
	ToolResultError bool `json:"tool_result_error,omitempty" yaml:"tool_result_error,omitempty"`

	// ExtraFields carries provider data that applies to the whole message. Block
	// replay data belongs in [Block.ExtraFields]. OpenAI Responses assistant
	// messages can use [ResponsesMessageExtraFieldPhase].
	ExtraFields map[string]interface{} `json:"extra_fields,omitempty" yaml:"extra_fields,omitempty"`
}

// Dialog is the ordered conversation supplied in [GenerationRequest.Dialog].
// System instructions are stored separately in [GenerationRequest.Instructions].
type Dialog []Message

// Str adapts a string to fmt.Stringer for use as [Block.Content].
type Str string

// String returns the underlying string.
func (s Str) String() string {
	return string(s)
}

// SystemMessage returns a [System]-role message containing a copy of blocks.
// Use it for [GenerationRequest.Instructions], not [GenerationRequest.Dialog].
func SystemMessage(blocks ...Block) Message {
	return Message{Role: System, Blocks: append([]Block(nil), blocks...)}
}

// TextBlock returns a [Content] block containing UTF-8 text with MIME type
// "text/plain".
func TextBlock(text string) Block {
	return Block{
		BlockType:    Content,
		ModalityType: Text,
		MimeType:     "text/plain",
		Content:      Str(text),
	}
}

// MetadataBlock encodes metadata as JSON in a [MetadataBlockType] block.
// [StreamingGenerator] implementations use this as their final successful data
// chunk; [StreamingAdapter] moves it to [Response.UsageMetadata]. Invalid map
// values produce an empty JSON object.
func MetadataBlock(metadata Metadata) Block {
	jsonData, err := json.Marshal(metadata)
	if err != nil {
		jsonData = []byte("{}")
	}

	return Block{
		BlockType:    MetadataBlockType,
		ModalityType: Text,
		MimeType:     "application/json",
		Content:      Str(jsonData),
	}
}

// ImageBlock returns an [Image] [Content] block by base64-encoding raw data.
// mimeType must identify the supplied bytes, for example "image/jpeg" or
// "image/png". Provider types document accepted image formats.
func ImageBlock(data []byte, mimeType string) Block {
	base64Data := base64.StdEncoding.EncodeToString(data)
	return Block{
		BlockType:    Content,
		ModalityType: Image,
		MimeType:     mimeType,
		Content:      Str(base64Data),
	}
}

// AudioBlock returns an [Audio] [Content] block by base64-encoding raw data.
// mimeType must identify the supplied bytes. Provider types document accepted
// audio formats.
func AudioBlock(data []byte, mimeType string) Block {
	base64Data := base64.StdEncoding.EncodeToString(data)
	return Block{
		BlockType:    Content,
		ModalityType: Audio,
		MimeType:     mimeType,
		Content:      Str(base64Data),
	}
}

// BlockFieldFilenameKey is the [Block.ExtraFields] key set by [PDFBlock]. Its
// string value is the caller-supplied filename used by providers that require a
// file name for document input.
const BlockFieldFilenameKey = "filename"

// PDFBlock returns an [Image] [Content] block with MIME type "application/pdf".
// It base64-encodes raw data and stores filename under [BlockFieldFilenameKey].
// PDF support and transport differ by provider; unsupported adapters return
// [UnsupportedInputModalityErr] or [InvalidParameterErr].
func PDFBlock(data []byte, filename string) Block {
	base64Data := base64.StdEncoding.EncodeToString(data)
	return Block{
		BlockType:    Content,
		ModalityType: Image,
		MimeType:     "application/pdf",
		Content:      Str(base64Data),
		ExtraFields: map[string]interface{}{
			BlockFieldFilenameKey: filename,
		},
	}
}

// ToolCallBlock returns a [ToolCall] block containing a JSON-encoded
// [ToolCallInput]. id is the provider call identifier, toolName must match a
// [Tool.Name], and parameters must be JSON-encodable.
func ToolCallBlock(id, toolName string, parameters map[string]any) (Block, error) {
	toolUse := ToolCallInput{
		Name:       toolName,
		Parameters: parameters,
	}

	data, err := json.Marshal(toolUse)
	if err != nil {
		return Block{}, fmt.Errorf("failed to marshal tool use: %w", err)
	}

	return Block{
		ID:           id,
		BlockType:    ToolCall,
		ModalityType: Text,
		MimeType:     "text/plain",
		Content:      Str(data),
	}, nil
}

// SeparatorBlock returns an internal [Separator] boundary. A
// [StreamingGenerator] can emit it between logical provider blocks;
// [StreamingAdapter] uses the boundary during compression and omits it from the
// final [Response].
func SeparatorBlock() Block {
	return Block{
		BlockType:    Separator,
		ModalityType: Text,
		MimeType:     "text/plain",
		Content:      Str(""),
	}
}

// ToolResultMessage returns a [ToolResult]-role message for the call id. It
// copies blocks and sets [Block.ID] on every copy, preserving the caller's
// blocks. Set [Message.ToolResultError] on the returned value when execution
// produced a model-visible tool error.
func ToolResultMessage(id string, blocks ...Block) Message {
	// Set the ID on all blocks
	resultBlocks := make([]Block, len(blocks))
	for i, block := range blocks {
		resultBlocks[i] = block
		resultBlocks[i].ID = id
	}
	return Message{
		Role:   ToolResult,
		Blocks: resultBlocks,
	}
}
