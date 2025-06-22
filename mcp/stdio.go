package mcp

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// StdioConfig contains configuration for stdio transport
type StdioConfig struct {
	// Timeout is the default timeout for operations
	Timeout int `json:"timeout,omitempty"`

	// Command to execute
	Command string `json:"command"`

	// Arguments for the command
	Args []string `json:"args"`

	// Environment variables to set
	Env map[string]string `json:"env,omitempty"`
}

// Stdio implements Transport using stdio communication.
// Thread Safety: This transport is NOT thread-safe. Connect/Close must not
// be called concurrently. Send/Receive operations are synchronized internally
// by the codec.
type Stdio struct {
	config    StdioConfig
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	stdout    io.ReadCloser
	stderr    io.ReadCloser
	stderrBuf *bufio.Reader
	codec     *Codec

	// State
	connectedState atomic.Bool
	closedState    atomic.Bool

	// For logging stderr
	onStderr func(string)

	// Channel for receiving messages
	receiveChan chan MessageOrError
	receiveOnce sync.Once
}

// NewStdio creates a new stdio transport
func NewStdio(config StdioConfig) *Stdio {
	return &Stdio{
		config: config,
	}
}

// SetStderrHandler sets a handler for stderr output
func (t *Stdio) SetStderrHandler(handler func(string)) {
	t.onStderr = handler
}

// Connect establishes the stdio transport connection.
func (t *Stdio) Connect(ctx context.Context) error {
	if t.connectedState.Load() {
		return ErrAlreadyConnected
	}

	// Create command
	t.cmd = exec.CommandContext(ctx, t.config.Command, t.config.Args...)

	// Set environment variables
	if len(t.config.Env) > 0 {
		env := os.Environ()
		for k, v := range t.config.Env {
			env = append(env, fmt.Sprintf("%s=%s", k, v))
		}
		t.cmd.Env = env
	}

	// Create pipes
	var err error
	t.stdin, err = t.cmd.StdinPipe()
	if err != nil {
		return NewTransportError("stdio", "create stdin pipe", err)
	}

	t.stdout, err = t.cmd.StdoutPipe()
	if err != nil {
		return NewTransportError("stdio", "create stdout pipe", err)
	}

	// TODO: figure out whether we even need a stderr pipe if the server is supposed to send logs to client
	t.stderr, err = t.cmd.StderrPipe()
	if err != nil {
		return NewTransportError("stdio", "create stderr pipe", err)
	}

	// Start the process
	if err := t.cmd.Start(); err != nil {
		return NewTransportError("stdio", "start process", err)
	}

	// Create codec
	t.codec = NewCodec(t.stdout, t.stdin)

	// Start stderr reader
	t.stderrBuf = bufio.NewReader(t.stderr)
	go t.readStderr()

	t.connectedState.Store(true)
	return nil
}

// readStderr reads and logs stderr output
func (t *Stdio) readStderr() {
	for {
		line, err := t.stderrBuf.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				// Log error if handler is set
				if t.onStderr != nil {
					t.onStderr(fmt.Sprintf("stderr read error: %v", err))
				}
			}
			return
		}

		if t.onStderr != nil && line != "" {
			t.onStderr(line)
		}
	}
}

// Close closes the stdio transport connection.
func (t *Stdio) Close() error {
	if t.closedState.Load() {
		return nil
	}
	t.closedState.Store(true)
	t.connectedState.Store(false)

	var errs []error

	// Close stdin to signal the process
	if t.stdin != nil {
		if err := t.stdin.Close(); err != nil {
			errs = append(errs, fmt.Errorf("close stdin: %w", err))
		}
	}

	// Wait for process to exit with timeout
	done := make(chan error, 1)
	go func() {
		if t.cmd != nil && t.cmd.Process != nil {
			done <- t.cmd.Wait()
		} else {
			done <- nil
		}
	}()

	select {
	case err := <-done:
		var exitErr *exec.ExitError
		if err != nil && errors.As(err, &exitErr) {
			errs = append(errs, fmt.Errorf("process wait: %w", exitErr))
		}
	case <-time.After(5 * time.Second):
		// Try SIGTERM first
		if t.cmd != nil && t.cmd.Process != nil {
			if err := t.cmd.Process.Signal(syscall.SIGTERM); err != nil {
				errs = append(errs, fmt.Errorf("send SIGTERM: %w", err))
			}

			// Give it another 2 seconds
			select {
			case <-done:
				// Process exited
			case <-time.After(2 * time.Second):
				// Force kill
				if err := t.cmd.Process.Kill(); err != nil {
					errs = append(errs, fmt.Errorf("kill process: %w", err))
				}
			}
		}
	}

	// Close pipes
	if t.stdout != nil {
		t.stdout.Close()
	}
	if t.stderr != nil {
		t.stderr.Close()
	}

	if len(errs) > 0 {
		return NewTransportError("stdio", "close", fmt.Errorf("multiple errors: %v", errs))
	}

	return nil
}

// GetProcessInfo returns information about the running process
func (t *Stdio) GetProcessInfo() (pid int, running bool) {
	if t.cmd != nil && t.cmd.Process != nil {
		pid = t.cmd.Process.Pid
		// Check if process is still running
		if err := t.cmd.Process.Signal(syscall.Signal(0)); err == nil {
			running = true
		}
	}

	return pid, running
}

// Send sends a JSON-RPC message.
// Thread-safe due to codec internal synchronization.
func (t *Stdio) Send(msg RpcMessage) error {
	if !t.connectedState.Load() {
		return ErrNotConnected
	}

	if t.codec == nil {
		return fmt.Errorf("codec not initialized")
	}

	return t.codec.WriteMessage(msg)
}

// Receive returns a channel that delivers messages or errors.
// The channel will be closed when the transport is closed.
func (t *Stdio) Receive() <-chan MessageOrError {
	// Ensure we only create the goroutine once
	t.receiveOnce.Do(func() {
		t.receiveChan = make(chan MessageOrError, 1)
		go t.receiveLoop()
	})
	return t.receiveChan
}

// receiveLoop continuously reads messages and sends them to the channel
func (t *Stdio) receiveLoop() {
	defer close(t.receiveChan)

	for {
		if !t.connectedState.Load() {
			return
		}

		if t.codec == nil {
			t.receiveChan <- MessageOrError{Error: fmt.Errorf("codec not initialized")}
			return
		}

		msg, err := t.codec.ReadMessage()
		if err != nil {
			if !t.closedState.Load() {
				t.receiveChan <- MessageOrError{Error: err}
			}
			return
		}

		t.receiveChan <- MessageOrError{Message: msg}
	}
}
