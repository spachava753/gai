package gai

import (
	"errors"
	"fmt"
)

// MaxGenerationLimitErr is returned when a Generator has generated the maximum number of tokens
// specified by GenOpts.MaxGenerationTokens. This error indicates that the generation was terminated
// due to reaching the token limit rather than natural completion.
var MaxGenerationLimitErr = errors.New("maximum generation limit exceeded")

// UnsupportedInputModalityErr is returned when a Generator encounters an input Message
// with a Block that contains an unsupported Modality. The string value of this error
// contains the name of the unsupported modality.
//
// For example, if a Generator only supports text input but receives an audio input,
// it will return this error with details about the unsupported audio modality.
type UnsupportedInputModalityErr string

func (u UnsupportedInputModalityErr) Error() string {
	return fmt.Sprintf("unsupported input modality: %s", string(u))
}

// UnsupportedOutputModalityErr is returned when a Generator is requested to generate
// a response in a Modality that it does not support via GenOpts.OutputModalities.
// The string value of this error contains the name of the unsupported modality.
//
// For example, if a Generator only supports text output but is asked to generate
// audio content, it will return this error with details about the unsupported
// audio modality.
type UnsupportedOutputModalityErr string

func (u UnsupportedOutputModalityErr) Error() string {
	return fmt.Sprintf("unsupported output modality: %s", string(u))
}

// InvalidToolChoiceErr is returned when an invalid tool choice is specified in
// GenOpts.ToolChoice. This can occur in several scenarios:
//   - When a specific tool is requested but doesn't exist
//   - When tools are required (ToolChoiceToolsRequired) but no tools are provided
//
// The string value of this error contains details about why the tool choice was invalid.
type InvalidToolChoiceErr string

func (i InvalidToolChoiceErr) Error() string {
	return fmt.Sprintf("invalid tool choice: %s", string(i))
}
