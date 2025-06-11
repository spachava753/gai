package mcp

import (
	"time"
)

// Options contains options for creating a client
type Options struct {
	// Timeout for requests
	RequestTimeout time.Duration

	// Handler for incoming messages
	MessageHandler func(msg interface{})

	// Handler for errors
	ErrorHandler func(err error)

	// Enable logging
	EnableLogging bool

	// Log handler
	LogHandler func(level, message string)
}

// DefaultOptions returns default client options
func DefaultOptions() Options {
	return Options{
		RequestTimeout: 30 * time.Second,
		MessageHandler: func(msg interface{}) {
			// Default: ignore unhandled messages
		},
		ErrorHandler: func(err error) {
			// Default: ignore errors
		},
	}
}
