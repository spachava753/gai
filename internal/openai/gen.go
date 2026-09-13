// Package openai contains the generated shared Chat Completions wire client.
// It is not yet used by gai.OpenAiGenerator.
//
// Use Client's raw HTTP response methods for streaming and full error fidelity.
// ClientWithResponses buffers and closes the body; it does not parse SSE.
// Callers must supply authentication through request editors and close raw
// response bodies. HTTP error statuses do not by themselves produce Go errors.
package openai

//go:generate go tool oapi-codegen -config config.yaml -o client.gen.go api.yaml
