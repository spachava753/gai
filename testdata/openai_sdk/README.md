# SDK request corpus

[`matrix.json`](matrix.json) supplies the conversation and SDK options once. The Go tests translate these inputs into public GAI options and compare the resulting HTTP request bodies with captured SDK requests. Each scenario runs through both `OpenAiGenerator` and the corresponding Z.AI, DeepSeek, or Moonshot wrapper (OpenAI cases use the OpenAI adapter in both modes). Explicit SDK options remain authoritative, including historical `max_tokens` usage on Moonshot. Separate local tests check wrapper defaults. Later requests use GAI's actual responses and tool calls, not copied expected messages.

Each `captures/*.json` file is an ordered array containing only `request` and `response` body strings. JSON and SSE bodies are preserved without parsing/reserializing them. HTTP compression is decoded; replay supplies HTTP 200 and the appropriate content type. No headers, decoded SDK outputs, tool results, or capture metadata are saved.

The corpus has **67 cases and 157 exchanges** across OpenAI (`gpt-5-mini` / `gpt-5.1`), Z.AI (`glm-5.3-flash`), Kimi (`kimi-k3`), and DeepSeek (`deepseek-flash`, the V4.1 Flash API alias) in JSON and SSE modes. It covers multi-turn tool use, image inputs/results, rich arguments, tool-choice controls, reasoning off, tool recovery, structured output, and PDF inputs where live recording succeeded.

## Latest-model recording results

The approved latest-model batch attempted **102 calls**, below its 106-call limit, with a 1,024-output-token cap and no retries. It produced **41 passing conversations / 97 exchanges**, all checked against GAI before adoption:

| Provider | Model | Passing cases | Adopted exchanges | Failed cases |
| --- | --- | ---: | ---: | ---: |
| Z.AI | `glm-5.3-flash` | 10 | 28 | 4 |
| Kimi | `kimi-k3` | 17 | 41 | 1 |
| DeepSeek | `deepseek-flash` | 14 | 28 | 0 |

GLM-5.3-Flash and Kimi K3 always reason, so their old reasoning-off profiles are excluded. Both use top-level `reasoning_effort: low`, mapped through GAI's `WithReasoningEffort`. GLM preserves thinking with `clear_thinking: false`; K3 omits the legacy `thinking` parameter. K3 named-function forcing is excluded because it is incompatible with thinking. DeepSeek's four earlier cases already use `deepseek-flash`; all fourteen previously failed conversations now have passing recordings. OpenAI captures are unchanged.

The active matrix contains no older Kimi or GLM profiles. Existing captures were replaced only with newly recorded body pairs, never relabeled. The previous corpus is backed up in `.plan/pre-latest-sdk-corpus`; synthetic smoke responses and failed conversations remain outside the active corpus.

Sources: [GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash), [Kimi K3](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart), [Kimi thinking parameters](https://platform.kimi.ai/docs/guide/use-thinking-models), [Kimi tool choice](https://platform.kimi.ai/docs/guide/use-tool-choice), [DeepSeek API model aliases](https://api-docs.deepseek.com/).

## Test offline

```sh
LIVE_TESTS= go test ./...
# Just the corpus and request comparator:
LIVE_TESTS= go test -count=1 -run 'TestOpenAISDKRequestBodies|TestSDKRequestHarness' .
```

Comparison uses JSON equivalence, including JSON inside tool-argument strings. Object ordering, JSON whitespace, and equivalent numeric notation do not matter. Array order, field presence, nulls, ordinary text, IDs, and parameter values do. Tool arguments must remain string-typed. These tests require neither Python nor credentials and never refresh captures.

Every replayed response must also finish successfully, return the expected number of usable function calls with unique IDs and schema-valid arguments, or produce a nonempty final answer. Scenarios with known arguments or JSON answers check those values semantically. These expectations are scenario facts, not stored SDK outputs. Reasoning-off profiles must not emit reasoning text.

## Additional feature coverage

The scenarios were constructed with the Python SDK first, checked locally against GAI, then recorded live through mitmproxy. Current passing coverage in both JSON and SSE modes:

| Coverage | Providers with passing captures |
| --- | --- |
| Rich nested arguments, optional omission, nulls, arrays, numbers, booleans, Unicode; two distinct tools; explicit auto selection | OpenAI, Kimi, DeepSeek |
| Same rich-tool conversation with reasoning off | OpenAI `gpt-5.1`, DeepSeek |
| Required tool | OpenAI, Kimi, DeepSeek |
| Named tool | OpenAI, DeepSeek |
| Forbidden tools | OpenAI, DeepSeek; Kimi in SSE mode only |
| Tool error followed by a corrected call and successful result | All four |
| JSON mode and multi-turn typed output | All four |
| JSON-schema output and multi-turn typed output | OpenAI, Kimi |
| Inline PDF and follow-up using PDF history | OpenAI |

Step-level `sdk_options` replace whole case parameters for that turn; for example, a forced tool selection returns to `auto` after the tool result. The Go runner maps tool choice through `WithToolChoice`, rather than bypassing the public API with raw request replacement. These tests caught and fixed `"none"` being incorrectly encoded as a named function.

**PDF scenarios are intentionally OpenAI-only**, in both JSON and SSE modes. Z.AI PDF cases have been removed from the pending and staged recording matrices; upload, OCR, and preprocessing workarounds are outside this corpus's scope. Historical experiment logs remain for reference, not as cases to retry.

`invoice.pdf` is a 652-byte synthetic one-page invoice with a known ID, vendor, and amount. It is sent inline using `file.filename` and `file.file_data`, not converted into text or images by the driver. The first response checks invoice facts as JSON. The follow-up checks the vendor name without requiring a JSON wrapper: the prompt asks for only the vendor, and OpenAI correctly answered `Acme Lab`. Both PDF captures were revalidated offline after correcting that assertion; their request/response bodies were not changed.

### Gaps observed during recording

**Follow-up diagnosis: PDF support works, but acceptance depends on processing state.** A fresh three-page PDF failed inline twice (15 seconds apart); after uploading the identical bytes with `purpose: "user_data"`, both `file_id` and the unchanged inline request returned its previously undisclosed random code correctly. The original 652-byte invoice and BinaryPC paper also became usable inline after upload; BinaryPC's arXiv URL then worked too. Upload-and-reference is the demonstrated workflow. `purpose: "agent"` uploads succeeded but their IDs failed in chat, including after a 30-second wait. GLM-OCR separately read BinaryPC's third page from arXiv. This strongly suggests a server-side preprocessing/cache dependency, not absent PDF capability; the exact backend cause and cache lifetime remain unknown. Rewriting was not established as necessary. See `.plan/glm-pdf-investigation.md` for the full route matrix, controls, and logs. All 10 returned upload IDs were deleted. These experiments did not add corpus fixtures or change production code; the failures below describe the original recording attempts.

Five latest-model cases were not adopted:

- **Z.AI `glm-5.3-flash` (2 rich-tool cases):** returned the string `"null"` instead of JSON `null` for `coupon`. This semantic mismatch was not normalized away.
- **Z.AI `glm-5.3-flash` (2 PDF cases):** HTTP 400, `Failed to parse the file. Please check its accessibility and format.` The tested inline PDF format failed during this recording, so the corpus does not claim passing GLM PDF coverage. Follow-up experiments described above demonstrated that the same invoice works inline after `user_data` upload.
- **Kimi `kimi-k3` (non-streaming forbidden-tools case):** returned `finish_reason: stop` with reasoning text but an empty final answer. The streaming case passed. Neither response is evidence of token-limit truncation; the non-streaming case fails the required usable-answer check.

A separate, explicitly approved two-call probe sent [*Attention Is All You Need*](https://arxiv.org/pdf/1706.03762) as `{"type":"file","file":{"file_url":"https://arxiv.org/pdf/1706.03762"}}` to GLM-5.3-Flash. Both JSON and SSE calls completed with `finish_reason: stop` and returned the correct title, first author, and Table 3 base-model values (`N=6`, `d_model=512`, `h=8`). No retries were used; the output cap was 1,024 tokens. This confirms acceptance of the PDF URL format, not that inline PDFs are categorically unsupported or that a familiar paper's answers prove retrieval. Raw pairs are in `.plan/glm-pdf-url-_qsv_oxi/bodies.json`. This SDK-only probe is not part of the differential corpus: GAI's current `PDFBlock` sends inline data, not `file_url`.

A follow-up probe used [*Tiny Pointers*](https://arxiv.org/pdf/2111.12800) and asked for details from the third physical PDF page: its printed page number/subsection heading, two storage overheads, the pointer-size tail bound, and citation numbers in the last paragraph. Local PDF extraction established the expected answers without sending them to GLM. Both the version-pinned `2111.12800v1` URL and canonical `2111.12800` URL failed in JSON and SSE modes: HTTP 400, code `1210`, `Failed to parse the file. Please check its accessibility and format.` Four calls were made, with SDK retries disabled. Both PDFs downloaded and parsed locally (34 pages); the provider error does not distinguish remote fetching from PDF parsing failure. Thus URL-based PDFs work for the Attention paper probe but are not established as reliable for arbitrary documents. No Tiny Pointers conversations were added to the corpus.

Two recent, specialized papers were then probed with third-page questions, using only their arXiv PDF URLs as document input. [BinaryPC](https://arxiv.org/abs/2608.04405) (August 5, 2026; 21 PDF pages) failed with HTTP 400/code `1210` in both JSON and SSE modes. [ASH](https://arxiv.org/abs/2606.07870) (June 5, revised August 24, 2026; 27 PDF pages) timed out after 180 seconds in JSON mode and returned HTTP 400/code `1210` in SSE mode. All four calls used a 1,024-output-token cap with retries disabled. Both PDFs downloaded and parsed locally. Citation counts could not be verified (Semantic Scholar returned HTTP 429); recency, not a claimed zero citation count, was the selection proxy. Questions targeted concrete page details such as Spotlight training duration/sample count and ASH's index citation numbers. No answers or local extracted text were supplied to GLM, and no conversations were adopted. Results are recorded in `.plan/recent-paper-urls.log`.

The previous batch's DeepSeek transport failures did not recur: all fourteen cases passed. The earlier GLM models also failed the rich-tool null and inline-PDF cases; switching to GLM-5.3-Flash did not resolve those observed failures.

Kimi's documented PDF workflow uploads and extracts text, and DeepSeek's Files API documents image formats only; those are not claimed as native PDF coverage. Z.AI documents only automatic tool choice, so forced/forbidden tool cases were not sent there. The three remaining unadopted, non-PDF case definitions are in `.plan/pending-sdk-matrix.json` for investigation with fresh live approval. Z.AI PDF cases are retired, not pending retries. Failure bookkeeping and synthetic smoke-test responses are not corpus fixtures.

Support references: [OpenAI file inputs](https://developers.openai.com/api/docs/guides/file-inputs), [Z.AI file inputs and tool-choice limits](https://docs.z.ai/api-reference/llm/chat-completion), [Kimi file Q&A](https://platform.kimi.ai/docs/guide/use-kimi-api-for-file-based-qa), [DeepSeek file limits](https://api-docs.deepseek.com/guides/files_api/), [Kimi tool choice](https://platform.kimi.ai/docs/guide/use-tool-choice), [Kimi structured output](https://platform.kimi.ai/docs/guide/response_format).

No cancellation, malformed-stream, or timeout scenarios were added to the batch.

## Image scenarios

The matrix also defines two scenarios using local files only:

- `user_images`: a JPEG in the first user turn, then ordered JPEG and PNG images in a follow-up that retains the earlier image and assistant reply.
- `tool_images`: a `get_image` call, a tool result containing text and a PNG under the returned call ID, then a text-only follow-up retaining that tool result. The tool returns a fixed red-square image; it does not read arbitrary model-supplied paths.

`sample.jpg` is the existing repository image; `shapes.png` is a small synthetic red square on white. `TestSDKRequestHarness/media_inputs` checks image bytes, MIME types, part ordering, tool-role IDs, and history serialization locally. All 28 image exchanges were captured from explicitly approved live SDK calls, without retries. Missing captures are errors, not silently skipped.

Live-confirmed Chat Completions coverage, in both JSON and SSE modes:

| Provider/model | User images | Image tool results |
| --- | --- | --- |
| OpenAI `gpt-5-mini` | Yes | Not in this Chat Completions corpus |
| DeepSeek `deepseek-flash` | Yes | No: its Chat Completions guide restricts images to user messages |
| Kimi `kimi-k3` | Yes | Yes |
| Z.AI `glm-5.3-flash` | Yes | Yes |

Sources: [OpenAI vision](https://developers.openai.com/api/docs/guides/images-vision), [DeepSeek vision restrictions](https://api-docs.deepseek.com/guides/vision/), [Kimi K3](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart), [GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash). Responses API tool-image support is a different contract and is not inferred to work on Chat Completions.

Cases select their scenario and model directly, for example `kimi/json/user_images` and `zai/stream/tool_images`. Z.AI uses `glm-5.3-flash` for text and image cases. After fresh live approval, with a Z.AI reverse proxy already running into a new output file:

```sh
uv run --python 3.11 --locked --script testdata/openai_sdk/conversation.py \
  zai/stream/tool_images
```

The runner now uses the OpenAI Python SDK for every provider, including Z.AI, to check supported API behavior rather than reproduce each SDK's serialization quirks. GAI images use data URLs; there is no bare-base64 option.

All recorded requests were made through the OpenAI SDK. Images preserve data URLs, including native tool-result images and retained history. All 157 request comparisons pass, without a bare-base64 option or image-URL normalization.

## Build the corpus

There are only two Python components:

- [`conversation.py`](conversation.py) runs one matrix case through the provider's pinned SDK. It maintains conversation history and executes only fixed synthetic price, image, order-quote, and shipping tools. It does not record traffic.
- [`capture.py`](capture.py) is a small [mitmproxy reverse-proxy addon](https://docs.mitmproxy.org/stable/concepts/modes/#reverse-proxy). Mitmproxy handles HTTP forwarding and buffering; the addon saves completed successful body pairs to a new file.

**Get explicit approval before making provider calls.** Start one proxy and one conversation per case. For example, to capture `zai/stream`, run this from the repository root in one terminal:

```sh
mkdir -p .plan/sdk-captures
uvx --from mitmproxy==12.2.3 mitmdump --quiet \
  --mode reverse:https://api.z.ai \
  --listen-host 127.0.0.1 --listen-port 8080 \
  --set confdir=.plan/mitmproxy \
  -s testdata/openai_sdk/capture.py \
  --set capture_file=.plan/sdk-captures/zai_stream.json
```

In another terminal, supply `Z_API_KEY` through your normal secret-management workflow, then run:

```sh
uv run --python 3.11 --locked --script testdata/openai_sdk/conversation.py \
  zai/stream --proxy http://127.0.0.1:8080
```

Use the selected case's `base_url` **origin** as the reverse-proxy target; the conversation script retains its API path prefix. Credentials come only from the case's `key_env`; the script does not invoke 1Password. The proxy listens only on loopback, so no interception certificate setup is needed. The SDK uses no retries, ignores environment proxies, and has a 180-second timeout.

Stop mitmdump after the conversation. Use a fresh output file for each case; existing files are refused. Do not enable mitmproxy body streaming or flow dumps: the addon needs buffered JSON/SSE bodies, and flow dumps would save headers and credentials. Bodies containing the bearer credential, non-200 responses, and SSE responses without `[DONE]` are not saved. Failed conversations may leave a partial array; discard it rather than resume it.

Review the body pairs and check their count against the scenario's steps before deliberately replacing that case's corpus file, then run the Go tests. Captured SDK requests are the baseline; there is no separate Python verification framework. Dependency downloads may be needed initially; `UV_OFFLINE=1` disables them once cached. The conversation's SDK dependencies are locked in `conversation.py.lock`.
