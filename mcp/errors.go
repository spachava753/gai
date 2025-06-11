// This file provides typed errors for the MCP client implementation.
package mcp

import (
	"errors"
	"fmt"
)

// Base error types
var (
	// ErrNotConnected is returned when operations are attempted on a disconnected client
	ErrNotConnected = errors.New("client not connected")

	// ErrAlreadyConnected is returned when Connect is called on an already connected client
	ErrAlreadyConnected = errors.New("client already connected")

	// ErrAlreadyInitialized is returned when Initialize is called on an already initialized client
	ErrAlreadyInitialized = errors.New("client already initialized")

	// ErrNotInitialized is returned when operations requiring initialization are attempted before initialization
	ErrNotInitialized = errors.New("client not initialized")

	// ErrClientClosing is returned when operations are attempted on a closing client
	ErrClientClosing = errors.New("client is closing")

	// ErrTimeout is returned when a request times out
	ErrTimeout = errors.New("request timeout")

	// ErrCancelled is returned when a request is cancelled
	ErrCancelled = errors.New("request cancelled")
)

// TransportError represents errors that occur at the transport layer
type TransportError struct {
	Transport string
	Operation string
	Err       error
}

func (e *TransportError) Error() string {
	return fmt.Sprintf("%s transport error during %s: %v", e.Transport, e.Operation, e.Err)
}

func (e *TransportError) Unwrap() error {
	return e.Err
}

// NewTransportError creates a new transport error
func NewTransportError(transport, operation string, err error) error {
	return &TransportError{
		Transport: transport,
		Operation: operation,
		Err:       err,
	}
}

// ProtocolError represents errors in the protocol layer
type ProtocolError struct {
	Code    int
	Message string
	Data    interface{}
}

func (e *ProtocolError) Error() string {
	if e.Data != nil {
		return fmt.Sprintf("protocol error %d: %s (data: %v)", e.Code, e.Message, e.Data)
	}
	return fmt.Sprintf("protocol error %d: %s", e.Code, e.Message)
}

// Standard protocol error codes
const (
	CodeParseError     = -32700
	CodeInvalidRequest = -32600
	CodeMethodNotFound = -32601
	CodeInvalidParams  = -32602
	CodeInternalError  = -32603
)

// NewProtocolError creates a new protocol error
func NewProtocolError(code int, message string, data interface{}) error {
	return &ProtocolError{
		Code:    code,
		Message: message,
		Data:    data,
	}
}

// AuthenticationError represents authentication failures
type AuthenticationError struct {
	Reason string
}

func (e *AuthenticationError) Error() string {
	return fmt.Sprintf("authentication failed: %s", e.Reason)
}

// NewAuthenticationError creates a new authentication error
func NewAuthenticationError(reason string) error {
	return &AuthenticationError{Reason: reason}
}

// RateLimitError represents rate limiting errors
// TODO: refactor to store headers and response instead of a retry after value and message
type RateLimitError struct {
	RetryAfter int // seconds
	Message    string
}

func (e *RateLimitError) Error() string {
	if e.RetryAfter > 0 {
		return fmt.Sprintf("rate limit exceeded (retry after %d seconds): %s", e.RetryAfter, e.Message)
	}
	return fmt.Sprintf("rate limit exceeded: %s", e.Message)
}

// NewRateLimitError creates a new rate limit error
func NewRateLimitError(retryAfter int, message string) error {
	return &RateLimitError{
		RetryAfter: retryAfter,
		Message:    message,
	}
}

// ValidationError represents validation failures
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	if e.Field != "" {
		return fmt.Sprintf("validation error in field '%s': %s", e.Field, e.Message)
	}
	return fmt.Sprintf("validation error: %s", e.Message)
}

// NewValidationError creates a new validation error
func NewValidationError(field, message string) error {
	return &ValidationError{
		Field:   field,
		Message: message,
	}
}

// UnsupportedFeatureError represents attempts to use unsupported features
type UnsupportedFeatureError struct {
	Feature string
	Reason  string
}

func (e *UnsupportedFeatureError) Error() string {
	if e.Reason != "" {
		return fmt.Sprintf("feature '%s' not supported: %s", e.Feature, e.Reason)
	}
	return fmt.Sprintf("feature '%s' not supported", e.Feature)
}

// NewUnsupportedFeatureError creates a new unsupported feature error
func NewUnsupportedFeatureError(feature, reason string) error {
	return &UnsupportedFeatureError{
		Feature: feature,
		Reason:  reason,
	}
}

// VersionMismatchError represents protocol version incompatibilities
type VersionMismatchError struct {
	ClientVersion string
	ServerVersion string
}

func (e *VersionMismatchError) Error() string {
	return fmt.Sprintf("protocol version mismatch: client supports %s, server supports %s",
		e.ClientVersion, e.ServerVersion)
}

// NewVersionMismatchError creates a new version mismatch error
func NewVersionMismatchError(clientVersion, serverVersion string) error {
	return &VersionMismatchError{
		ClientVersion: clientVersion,
		ServerVersion: serverVersion,
	}
}

// Helper functions for error checking

// IsTransportError checks if an error is a TransportError
func IsTransportError(err error) bool {
	var te *TransportError
	return errors.As(err, &te)
}

// IsProtocolError checks if an error is a ProtocolError
func IsProtocolError(err error) bool {
	var pe *ProtocolError
	return errors.As(err, &pe)
}

// IsAuthenticationError checks if an error is an AuthenticationError
func IsAuthenticationError(err error) bool {
	var ae *AuthenticationError
	return errors.As(err, &ae)
}

// IsRateLimitError checks if an error is a RateLimitError
func IsRateLimitError(err error) bool {
	var re *RateLimitError
	return errors.As(err, &re)
}

// IsValidationError checks if an error is a ValidationError
func IsValidationError(err error) bool {
	var ve *ValidationError
	return errors.As(err, &ve)
}

// IsUnsupportedFeatureError checks if an error is an UnsupportedFeatureError
func IsUnsupportedFeatureError(err error) bool {
	var ue *UnsupportedFeatureError
	return errors.As(err, &ue)
}

// IsVersionMismatchError checks if an error is a VersionMismatchError
func IsVersionMismatchError(err error) bool {
	var ve *VersionMismatchError
	return errors.As(err, &ve)
}
