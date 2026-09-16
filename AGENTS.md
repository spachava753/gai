# GAI (Go AI generation utilities)

GAI is a Go library for interacting with LLM providers, including OpenAI, OpenCode, Anthropic, Google Gemini, Cerebras, OpenRouter, ZAI, DeepSeek, and Moonshot, with consistent APIs, composable generators, streaming support, metrics, and robust testing. The core library is in the repository root as a standard Go module.

@README.md provides a general overview and quick start. See @ROADMAP.md for planned work.

## Project structure and organization

- Root Go module (go.mod) with all library code colocated for ease of import
  - Provider clients: `openai.go`, `opencode.go`, `anthropic.go`, `gemini.go`, `cerebras.go`, `openrouter.go`, `responses.go`, `zai.go`, `deepseek.go`, `moonshot.go`
  - Z.AI, DeepSeek, and Moonshot delegate to private `*OpenAiGenerator` fields; each provider owns its request translation, defaults, and Generate/Stream delegation in its own file. Keep provider-specific logic out of `compatible.go`, which contains only shared error relabeling and JSON HTTP mechanics. Do not embed `GeneratorWrapper` or expose OpenAI's local counter. Z.AI and Moonshot use native HTTP token counters; Moonshot rejects tools in Count because tool accounting is undocumented. DeepSeek has no Count method. Provider wrappers use shared OpenAI wire/usage metadata keys and preserve explicit request options without model-name heuristics. Z.AI/DeepSeek default to `max_tokens`; Moonshot uses `max_completion_tokens`. Z.AI omits stream usage requests by default. Native option helpers are in provider files and `openai_options.go`.
  - `GenerationRequest.SafetyIdentifier` maps to documented provider end-user identity fields; `PromptCacheKey` maps only to OpenAI Chat Completions/Responses and Moonshot cache grouping. Unsupported providers ignore these fields. Never derive one from the other or map them to Gemini safety settings or cachedContent. `WithReasoningEffort(string)` is distinct from `WithThinkingBudget(int)`; Anthropic/OpenRouter support the numeric budget and reject combining the two.
  - Generated OpenAPI clients: `internal/openai/` uses oapi-codegen; `internal/cerebras/`, `internal/deepseek/`, `internal/opencode/`, `internal/openrouter/`, and `internal/zai/` still use OGEN (the DeepSeek/Z.AI clients are no longer used by their public adapters).
  - Shared Chat Completions client: `internal/openai/api.yaml`, `config.yaml`, and `oapi-overlay.yaml` generate the checked-in `client.gen.go` with lossless opaque JSON, nullable values, and float64 numbers. Run `go generate ./internal/openai`; generation paths are package-relative. `go test ./internal/openai` checks regeneration and local codec/HTTP contracts. Its raw HTTP methods back `OpenAiGenerator`; streaming framing and per-invocation assembly live in `openai_stream.go`, with no automatic POST retries.
  - SDK request contracts: `testdata/openai_sdk/matrix.json` declares named scenarios and SDK options once; cases select a scenario. The corpus has 67 cases / 157 exchanges across OpenAI, GLM-5.3-Flash, Kimi K3, and DeepSeek V4.1 Flash (`deepseek-flash`). It includes multi-turn tools, user images, native Kimi/Z.AI tool-result images, rich arguments, tool-choice controls, recovery, structured output, and OpenAI inline PDFs. All recorded request comparisons pass. The runner uses the OpenAI Python SDK for every provider, and GAI always sends image data URLs. Do not add production options just to reproduce SDK-specific serialization quirks. `captures/` holds only raw request/response body pairs. `TestOpenAISDKRequestBodies` in external-package `openai_sdk_test.go` exercises both the shared OpenAI generator and provider wrappers, translates SDK settings into public GAI options and checks JSON-equivalent HTTP bodies with isolated provider/mode/step cases and GAI-derived history, plus successful completion, usable tool calls, scenario arguments/JSON answers, and reasoning-off behavior. K3/GLM always-thinking profiles exclude reasoning-off cases; K3 named-tool forcing is excluded. The latest approved batch used 102 calls without retries and adopted 41 conversations / 97 exchanges. Three cases remain pending: GLM string-versus-null tool arguments (2) and K3 non-streaming forbidden-tools returning an empty final answer (1). PDF scenarios are intentionally OpenAI-only. Z.AI PDF cases were removed from pending/staged recording matrices; do not reintroduce upload, OCR, or preprocessing workarounds into this corpus without a new request. Historical PDF experiment artifacts are reference material, not pending cases. Follow-up PDF experiments found a processing-state dependency: fresh inline PDFs failed, then identical bytes succeeded after upload with `purpose: "user_data"`; `file_id` also worked. The original invoice and BinaryPC paper now have successful SDK-only probes, not corpus fixtures. `purpose: "agent"` file IDs failed in chat; GLM-OCR is a working separate extraction route. Do not claim PDF support is absent or require rewriting: the exact backend cause is unconfirmed. See `.plan/glm-pdf-investigation.md` and the corpus README. All 10 returned experimental upload IDs were deleted. All fourteen previously failing DeepSeek cases now pass. See the corpus README and `.plan/pending-sdk-matrix.json`; never normalize away failures or relabel old captures as newer-model evidence. Older Kimi/GLM profiles are retired from the active corpus, with the previous corpus backed up in `.plan/pre-latest-sdk-corpus`. The 652-byte `invoice.pdf` is synthetic; extraction/upload workflows are not native PDF coverage. `conversation.py` runs one SDK conversation through mitmproxy; `capture.py` saves body pairs. See its README for explicitly approved corpus-building commands. No separate Python verification framework or resume machinery. Never refresh captures automatically during tests.
  - Generation pipeline and composition: `generate.go`, `retry_generator.go`, `fallback_generator.go`, `preprocessing_generator.go`
  - Streaming primitives: `streaming.go`
  - Shared domain types and helpers: `message.go`, `tool.go`, `errors.go`, `metrics.go`, `callback.go`
  - Examples and usage via `*_example_test.go` files
- Public agent packages: `agent/` contains the reusable agent loop and `agent/agenttest/` contains deterministic test fixtures; `agent/design.md` documents behavior and rationale
- Tests: colocated `*_test.go` for each area, plus provider-specific tests
- Samples: `sample.jpg`, `sample.pdf`, `sample.wav` for multimodal tests/examples
- Tracked hooks: `.githooks/pre-commit` runs LAAS against hand-written packages and excludes generated client packages
- Public documentation: `doc.go` contains the package API map, `README.md` contains the repository guide, and `*_example_test.go` contains compiled examples
- `design.md` records the generator interfaces, shared types, state ownership, and rationale implemented by the current release
- `agent/design.md` records the implemented agent-loop API, run flow, state responsibilities, and rationale

Conventions
- Single module, no internal/ submodules yet
- Provider-neutral generation APIs live at the repository root in package `gai`; agent APIs live in `agent/`
- Experimental/stability-in-flux code goes under `x/`

## Build, test, and development commands

Requirements: Go 1.26.6+.

Common commands
- Install deps: `go mod download`
- Lint: `go tool laas -exclude-packages='^github\.com/spachava753/gai/internal/(cerebras|deepseek|openai|opencode|openrouter|zai)$' ./...`
- Lint with golangci-lint if installed: `golangci-lint run` (optional)
- Run tests (all, live API tests skipped): `go test ./...`
- Run live API tests only when explicitly requested by the user: `LIVE_TESTS=1 go test ./...` (also requires the relevant provider API keys)
- Run tests with race: `go test -race ./...`
- Run a single test file: `go test -run TestName ./...`
- Examples as docs: `go test ./...` executes `*_example_test.go`
- Inspect rendered package documentation: `go doc -all .`

Dev tips
- Use `RG_COLOR=never` if your environment requires plain output
- Prefer ripgrep (rg) to locate call sites quickly, e.g. `rg 'Generate\('`

## Code style and conventions

Go style
- Follow standard Go formatting: `gofmt`/`go fmt` and `go vet` clean
- Keep package surface minimal and clear; prefer small, composable types
- Error handling: return `error` values, wrap with `%w` using `fmt.Errorf`
- Context-first: public methods that block should accept `context.Context`
- Naming: exported identifiers use full words; keep acronyms consistent (ID, URL, API)
- Avoid panics in library code; prefer errors
- Keep provider-specific types in their files to prevent cross-coupling
- Use table-driven tests for variations

Documentation
- Keep `doc.go` concise and focused on provider-neutral concepts, with Go documentation links such as `[GenerationRequest]`
- Maintain @README.md as the repository guide; it is independent from `doc.go`
- Put provider-specific behavior on the corresponding provider type, constructor, option, metadata key, or method
- Public comments should explain intent, contracts, value types, storage locations, errors, and related symbols instead of restating declarations
- Use reciprocal links between option helpers and keys, and between metadata keys and their containing fields
- Example-driven documentation via `*_example_test.go`; examples must compile under `go test ./...`

Imports
- Standard -> third-party -> local groupings
- Avoid unnecessary type aliases; prefer direct types

## Architecture and design patterns

Key concepts
- Generator abstraction: pluggable components that transform inputs to model calls and outputs
- Composition patterns: retry, fallback, and preprocessing generators compose functionality without duplication
- Provider adapters: thin wrappers that expose a unified interface over provider SDKs, generated clients, and HTTP APIs
- Streaming: unified stream interface emitting chunks, with helpers for incremental assembly
- Tools: typed function/tool-call support with validation and safe dispatch
- Agent loop: package `agent` provides a reusable Agent with fixed borrowed configuration, one new user message per run, typed dialog/generation/tool hooks, borrowed ordered observer events, hook-controlled stopping, lasting dialog replacement, partial error results, and external persistence
- Metrics and callbacks: hooks for observability, tracing, and policy checks

Design principles
- Interface-first, implementation-behind adapters
- Small, testable units with clear contracts
- Provider-agnostic core, provider-specific edges
- Keep generated clients under `internal/`; public constructors accept standard HTTP clients, base URLs, and credentials rather than generated types
- Require callers to pass credentials explicitly; provider constructors do not read environment variables or other global configuration
- Opt into features (metrics, callbacks) without forcing dependencies

Provider-specific metadata placement
- Prefer `ExtraFields` for provider-specific data that is not universal across generators.
- Store provider metadata at the narrowest scope that matches the provider API contract:
  - Use `Message.ExtraFields` for message-level metadata.
  - Use `Block.ExtraFields` for block-level metadata.

## Testing guidelines

- Use `go test ./...` locally and in CI
- Structure tests as table-driven where relevant
- Prefer live tests, that is, actually making network calls with SDKs and our packages, mocks cannot provide the same testing gaurantees
- Gate every test that makes a live API call with `requireLiveAPIKey`; live tests run only when `LIVE_TESTS` and the relevant provider API key are both set
- Never run live tests unless the user explicitly asks for them, even when `LIVE_TESTS` and provider credentials are already set in the environment
- Example tests (`*_example_test.go`) should compile and run as documentation
- For streaming, assert on ordered chunk assembly and termination conditions
- Include negative tests: timeouts, API errors, invalid tool payloads
- Use sample media files for multimodal inputs to avoid external fetches
- Use `agent/agenttest` scripted generators and recording observers for deterministic Agent tests

Naming and location
- Unit tests live alongside implementation: `file_test.go`
- Minimize test file count: add tests to an existing relevant `*_test.go` file; create another test file only when no suitable file exists or technical separation requires it
- Example files: `*_example_test.go` demonstrate canonical usage

## Security considerations

Secrets
- Never commit API keys. Applications may load provider credentials from environment variables, then must pass them explicitly to constructors:
  - OpenAI: `OPENAI_API_KEY`
  - OpenCode: `OPENCODE_API_KEY`
  - Anthropic: `ANTHROPIC_API_KEY`
  - Google: `GOOGLE_API_KEY` or ADC where applicable
  - Cerebras: `CEREBRAS_API_KEY`
  - OpenRouter: `OPENROUTER_API_KEY`
  - ZAI: `Z_API_KEY`
  - Moonshot: `MOONSHOT_API_KEY`
  - DeepSeek: `DEEPSEEK_API_KEY`
- Provide `.env` locally but do not commit secrets; prefer `.env.example` when introduced

Data handling
- Avoid logging sensitive prompts or completions by default
- Redact potentially sensitive tool inputs in logs/metrics callbacks
- Validate tool-call payloads before execution
- Enforce timeouts and context cancellation for remote calls
- Follow provider usage policies and rate limits; implement exponential backoff in retry generators

Dependencies
- Keep dependencies minimal; regularly `go get -u` and review changelogs

## Testing frameworks and execution

- Standard library `testing` is used; no external test framework required
- Use `-run`, `-bench`, `-count=1` flags to target tests and disable caching when needed
- For race detection and leak checks: `go test -race ./...`

## Configuration

Environment variables
- Applications and live tests may load provider keys as listed above, but constructors require those keys as explicit arguments
- `LIVE_TESTS=1` explicitly enables network-backed tests; the relevant provider API key must also be set
- Tuning flags when present should be wired through options structs; document defaults on the public option symbol and in README when they affect common usage

Configuration management
- Prefer functional options to long parameter lists for public constructors
- Validate config at construction time; return descriptive errors

When adding new configuration
1) Add fields to the appropriate options or config struct
2) Validate in constructor
3) Document the public symbols and add an example test; update @README.md when the configuration belongs in the repository guide

## Git and contribution workflow

- Activate the tracked pre-commit hook after cloning: `git config --local core.hooksPath .githooks`
- This is a greenfield project. Work directly on `main` and push commits to `main` by default; feature branches and pull requests are not required unless explicitly requested.
- Keep commits small, focused, and with descriptive messages
- This module is pre-`v1.0.0`; breaking API changes are acceptable when they improve the package, though they should still be intentional and documented in code/tests/docs as appropriate
- Ensure `go fmt`, `go vet`, and `go test ./...` pass before pushing
- Avoid force pushes on `main`; use `--force-with-lease` only on feature branches

## Development environment

- Go 1.26.6+
- LAAS is pinned as a Go tool and runs through the tracked pre-commit hook
- Optional tools: `golangci-lint`, `rg` (ripgrep)

## Parsing notes for agentic tools

- Section headers are stable and use H2 for primary topics
- File references use @-mentions (e.g., @README.md, @ROADMAP.md)
- Commands appear in inline code blocks with backticks

## Documentation for Go Symbols

When gathering context about symbols like types, global variables, constants, functions and methods, prefer to use `go doc` command. You may use `go doc -all github.com/example/pkg` to get a full overview of a package, but use sparingly, as it may overwhelm the context window. Alternatively, you may use `go doc github.com/example/pkg.Type` to get documentation about a specific symbol. Try to avoid usage of `rg` cli to search over source code of dependencies, as it can easily overwhelm your context window.
