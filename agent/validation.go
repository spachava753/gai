package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"github.com/spachava753/gai"
)

var standardUsageKeys = [...]string{
	gai.UsageMetricInputTokens,
	gai.UsageMetricGenerationTokens,
	gai.UsageMetricCacheReadTokens,
	gai.UsageMetricCacheWriteTokens,
	gai.UsageMetricReasoningTokens,
}

// isNilInterface reports nil interfaces and interfaces containing a nil pointer
// or function implementation. Other nilable concrete values may have valid
// methods and are not treated as absent.
func isNilInterface(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	return (v.Kind() == reflect.Pointer || v.Kind() == reflect.Func) && v.IsNil()
}

func validateInstructions(message gai.Message) error {
	if message.Role == gai.RoleUnknown {
		if len(message.Blocks) == 0 && len(message.ExtraFields) == 0 && !message.ToolResultError {
			return nil
		}
		return errors.New("a populated instruction message must specify the system role")
	}
	if message.Role != gai.System {
		return fmt.Errorf("instructions must use the system role, got %s", message.Role)
	}
	if message.ToolResultError {
		return errors.New("instructions cannot be marked as a tool-result error")
	}
	if len(message.Blocks) == 0 {
		return nil
	}
	return validateMessage(message)
}

func validateDialog(dialog gai.Dialog) error {
	for i, message := range dialog {
		if message.Role == gai.System {
			return fmt.Errorf("dialog message %d uses the system role", i)
		}
		if err := validateMessage(message); err != nil {
			return fmt.Errorf("dialog message %d: %w", i, err)
		}
		if message.Role == gai.ToolResult {
			for j, block := range message.Blocks {
				if block.ID == "" {
					return fmt.Errorf("dialog message %d block %d: tool-result block ID is required", i, j)
				}
			}
		}
	}
	return nil
}

func validateMessage(message gai.Message) error {
	if message.ToolResultError && message.Role != gai.ToolResult {
		return errors.New("tool-result error is set on a non-tool-result message")
	}

	switch message.Role {
	case gai.User, gai.Assistant, gai.ToolResult, gai.System:
	default:
		return fmt.Errorf("unsupported message role %s", message.Role)
	}
	if len(message.Blocks) == 0 {
		return fmt.Errorf("%s message must contain at least one block", message.Role)
	}

	for i, block := range message.Blocks {
		if err := validateBlock(block, message.Role); err != nil {
			return fmt.Errorf("block %d: %w", i, err)
		}
	}
	return nil
}

// validateBlock enforces the block-type, role, content, and tool-call rules the
// loop depends on.
func validateBlock(block gai.Block, role gai.Role) error {
	if isNilInterface(block.Content) {
		return errors.New("content is required")
	}

	switch block.BlockType {
	case gai.Content:
		return nil
	case gai.Thinking:
		if role != gai.Assistant {
			return errors.New("thinking block requires the assistant role")
		}
		if block.ModalityType != gai.Text {
			return errors.New("thinking block must use text modality")
		}
		return nil
	case gai.ToolCall:
		if role != gai.Assistant {
			return errors.New("tool-call block requires the assistant role")
		}
		if block.ModalityType != gai.Text {
			return errors.New("tool-call block must use text modality")
		}
		if block.ID == "" {
			return errors.New("tool-call block ID is required")
		}
		return nil
	default:
		return fmt.Errorf("unsupported block type %q", block.BlockType)
	}
}

func validateOptions(options gai.GenerationOptions) error {
	value, ok := options[gai.GenerationOptionCandidateCount]
	if !ok {
		return nil
	}
	count, ok := value.(uint)
	if !ok {
		return gai.InvalidParameterErr{
			Parameter: gai.GenerationOptionCandidateCount,
			Reason:    fmt.Sprintf("expected uint, got %T", value),
		}
	}
	if count != 1 {
		return gai.InvalidParameterErr{
			Parameter: gai.GenerationOptionCandidateCount,
			Reason:    "agent runs require exactly one candidate",
		}
	}
	return nil
}

func prepareExecutableTools(tools []Tool) ([]executableTool, []gai.Tool, error) {
	prepared := make([]executableTool, 0, len(tools))
	definitions := make([]gai.Tool, 0, len(tools))
	seen := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		definition := tool.Definition
		if definition.Name == "" {
			return nil, nil, &gai.InvalidToolErr{Tool: definition.Name, Cause: errors.New("tool name cannot be empty")}
		}
		if definition.Name == gai.ToolChoiceAuto || definition.Name == gai.ToolChoiceToolsRequired {
			return nil, nil, &gai.InvalidToolErr{Tool: definition.Name, Cause: fmt.Errorf("tool name cannot be %s", definition.Name)}
		}
		if _, ok := seen[definition.Name]; ok {
			return nil, nil, &gai.InvalidToolErr{Tool: definition.Name, Cause: errors.New("tool already provided")}
		}
		if isNilInterface(tool.Handler) {
			return nil, nil, &gai.InvalidToolErr{Tool: definition.Name, Cause: errors.New("handler is required")}
		}
		seen[definition.Name] = struct{}{}

		preparedTool := executableTool{definition: definition, handler: tool.Handler}
		if definition.InputSchema != nil {
			resolved, err := definition.InputSchema.Resolve(nil)
			if err != nil {
				return nil, nil, &gai.InvalidToolErr{Tool: definition.Name, Cause: fmt.Errorf("resolve input schema: %w", err)}
			}
			preparedTool.schema = resolved
		}
		prepared = append(prepared, preparedTool)
		definitions = append(definitions, definition)
	}
	return prepared, definitions, nil
}

func toolDefinitionsEqual(left, right []gai.Tool) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if !reflect.DeepEqual(left[i], right[i]) {
			return false
		}
	}
	return true
}

func validateGenerationRequest(request gai.GenerationRequest, fixedTools []gai.Tool) error {
	if request.Model == "" {
		return gai.InvalidParameterErr{Parameter: "model", Reason: "cannot be empty"}
	}
	if err := validateInstructions(request.Instructions); err != nil {
		return gai.InvalidParameterErr{Parameter: "instructions", Reason: err.Error()}
	}
	if len(request.Dialog) == 0 {
		return gai.InvalidParameterErr{Parameter: "dialog", Reason: gai.ErrEmptyDialog.Error()}
	}
	if err := validateDialog(request.Dialog); err != nil {
		return gai.InvalidParameterErr{Parameter: "dialog", Reason: err.Error()}
	}
	if !toolDefinitionsEqual(request.Tools, fixedTools) {
		return gai.InvalidParameterErr{Parameter: "tools", Reason: "must exactly match the Agent's fixed tools"}
	}
	return validateOptions(request.Options)
}

func validateFinishReason(reason gai.FinishReason) error {
	switch reason {
	case gai.Unknown, gai.EndTurn, gai.StopSequence, gai.MaxGenerationLimit, gai.ToolUse, gai.ContentPolicyViolation:
		return nil
	default:
		return fmt.Errorf("unsupported finish reason %d", reason)
	}
}

// validateResponse checks the single assistant candidate, provider facts, and call-ID rules.
func validateResponse(response gai.Response, seenCallIDs map[string]struct{}) error {
	if len(response.Candidates) != 1 {
		return fmt.Errorf("expected one candidate, got %d", len(response.Candidates))
	}
	message := response.Candidates[0]
	if message.Role != gai.Assistant {
		return fmt.Errorf("candidate must use the assistant role, got %s", message.Role)
	}
	if err := validateMessage(message); err != nil {
		return fmt.Errorf("candidate: %w", err)
	}
	if err := validateFinishReason(response.FinishReason); err != nil {
		return err
	}
	if err := validateStandardUsage(response.UsageMetadata); err != nil {
		return fmt.Errorf("usage: %w", err)
	}

	toolCalls := 0
	responseIDs := make(map[string]struct{})
	for _, block := range message.Blocks {
		if block.BlockType != gai.ToolCall {
			continue
		}
		toolCalls++
		if _, ok := responseIDs[block.ID]; ok {
			return fmt.Errorf("duplicate tool-call ID %q in response", block.ID)
		}
		if _, ok := seenCallIDs[block.ID]; ok {
			return fmt.Errorf("duplicate tool-call ID %q in run", block.ID)
		}
		responseIDs[block.ID] = struct{}{}
	}
	if response.FinishReason == gai.ToolUse && toolCalls == 0 {
		return errors.New("tool-use finish reason requires a tool-call block")
	}
	if response.FinishReason != gai.ToolUse && toolCalls != 0 {
		return fmt.Errorf("tool-call blocks conflict with finish reason %d", response.FinishReason)
	}
	return nil
}

func validateAfterGenerationResponse(original, changed gai.Response, seenCallIDs map[string]struct{}) error {
	if original.FinishReason != changed.FinishReason {
		return errors.New("AfterGeneration must preserve FinishReason")
	}
	if !reflect.DeepEqual(original.UsageMetadata, changed.UsageMetadata) {
		return errors.New("AfterGeneration must preserve UsageMetadata")
	}
	return validateResponse(changed, seenCallIDs)
}

func decodeToolCall(block gai.Block) (gai.ToolCallInput, error) {
	var call gai.ToolCallInput
	if err := json.Unmarshal([]byte(block.Content.String()), &call); err != nil {
		return gai.ToolCallInput{}, fmt.Errorf("decode tool call: %w", err)
	}
	if call.Name == "" {
		return gai.ToolCallInput{}, errors.New("tool-call name is required")
	}
	return call, nil
}

func validateToolArguments(tool executableTool, parameters map[string]any) error {
	if tool.schema == nil {
		if len(parameters) != 0 {
			return fmt.Errorf("arguments for tool %q: tool declares no parameters", tool.definition.Name)
		}
		return nil
	}
	if err := tool.schema.Validate(parameters); err != nil {
		return fmt.Errorf("arguments for tool %q: %w", tool.definition.Name, err)
	}
	return nil
}

func validateToolResult(message gai.Message) error {
	if message.Role != gai.ToolResult {
		return fmt.Errorf("tool handler result must use the tool-result role, got %s", message.Role)
	}
	return validateMessage(message)
}

func validateStopReason(reason StopReason) error {
	if reason == StopReasonModel {
		return errors.New("model stop reason is reserved for terminal model responses")
	}
	return nil
}

func validateStandardUsage(metadata gai.Metadata) error {
	for _, key := range standardUsageKeys {
		value, ok := metadata[key]
		if !ok {
			continue
		}
		count, ok := value.(int)
		if !ok {
			return fmt.Errorf("metric %q must be int, got %T", key, value)
		}
		if count < 0 {
			return fmt.Errorf("metric %q cannot be negative", key)
		}
	}
	return nil
}

func addStandardUsage(total gai.Metadata, metadata gai.Metadata) (gai.Metadata, error) {
	if err := validateStandardUsage(metadata); err != nil {
		return total, err
	}
	maxInt := int(^uint(0) >> 1)
	hasValues := false
	for _, key := range standardUsageKeys {
		value, ok := metadata[key]
		if !ok {
			continue
		}
		hasValues = true
		current, _ := total[key].(int)
		if value.(int) > maxInt-current {
			return total, fmt.Errorf("metric %q overflows int", key)
		}
	}
	if !hasValues {
		return total, nil
	}
	if total == nil {
		total = make(gai.Metadata)
	}
	for _, key := range standardUsageKeys {
		if value, ok := metadata[key]; ok {
			current, _ := total[key].(int)
			total[key] = current + value.(int)
		}
	}
	return total, nil
}
