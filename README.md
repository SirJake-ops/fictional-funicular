# LLM Inference Engine

C++20 HTTP service for local ONNX-based text generation.

The application exposes a small HTTP API, loads an ONNX model from the local `models/` directory, and runs token-by-token generation through ONNX Runtime with KV-cache reuse between decode steps inside a request.

## Status

This project is usable as a local development service, not a hardened production deployment.

Current constraints:

- The service is configured for a single local process and a fixed listen address in code.
- Model assets are expected to exist on local disk before startup.
- Request validation and error handling are basic.
- The tokenizer is intentionally simplified and byte-based, not a model-matched production tokenizer.

## Features

- `POST /generate` endpoint for text generation
- ONNX Runtime-backed inference
- KV-cache reuse during autoregressive decoding
- Project-root-relative model resolution
- Unit test coverage for routing and inference behavior

## Requirements

- Linux or a compatible environment with a C++20 toolchain
- CMake `3.20+`
- A C++20 compiler
- ONNX Runtime shared library for your platform
- A compatible ONNX model at `models/model.onnx`

## Runtime Assets

This repository does not vendor the ONNX Runtime shared library or the ONNX model.

Expected local files:

- `models/libonnxruntime.so`
- `models/libonnxruntime.so.1`
- `models/model.onnx`

The shared libraries may be symlinks to an extracted ONNX Runtime release.

Example:

```bash
mkdir -p models
ln -sfn /path/to/onnxruntime-linux-x64-1.24.4/lib/libonnxruntime.so.1.24.4 models/libonnxruntime.so
ln -sfn /path/to/onnxruntime-linux-x64-1.24.4/lib/libonnxruntime.so.1.24.4 models/libonnxruntime.so.1
```

## Repository Layout

```text
include/fictional_funicular/   public headers
src/                           application and library source
tests/                         unit tests
examples/                      experimental or scratch code
third_party/                   vendored dependencies
models/                        local runtime assets, not committed
build/                         generated build output
```

## Build

From the project root:

```bash
cmake -S . -B build
cmake --build build
```

Produced executable:

```bash
./build/LLM_Inference_Engine
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Run

Start the server from the project root:

```bash
./build/LLM_Inference_Engine
```

Default bind address:

- Host: `127.0.0.1`
- Port: `1234`

This is currently hardcoded in [main.cpp](/home/jake/cpp-projects/fictional-funicular/src/app/main.cpp).

## HTTP API

### `GET /hi`

Health-style test endpoint.

Response:

```text
Hello from the class
```

### `POST /generate`

Primary text-generation endpoint.

Request:

- Body: raw prompt text
- Query parameter: `max_tokens`
- Default `max_tokens`: `16`

Example:

```bash
curl -X POST "http://127.0.0.1:1234/generate?max_tokens=8" \
  --data "Hello, "
```

Example response:

```json
{
  "prompt": "Hello, ",
  "generated_text": "<tok:123><tok:456>",
  "response_text": "Hello, <tok:123><tok:456>",
  "prompt_token_ids": [72,101,108,108,111,44,32],
  "generated_token_ids": [123,456],
  "prompt_token_count": 7,
  "generated_token_count": 2,
  "cache_layers": 24,
  "cache_sequence_length": 9
}
```

Response fields:

- `prompt`: original request text
- `generated_text`: decoded generated tokens
- `response_text`: `prompt + generated_text`
- `prompt_token_ids`: encoded prompt token ids
- `generated_token_ids`: generated token ids
- `prompt_token_count`: prompt token count
- `generated_token_count`: generated token count
- `cache_layers`: number of cache layers tracked by the model wrapper
- `cache_sequence_length`: cached sequence length after generation

### `GET /run_model`

Legacy debug endpoint that performs a single inference step and returns the next token id plus a decoded representation.

This route is useful for debugging, not as the primary API.

### `GET /stop`

Local shutdown route used for simple testing flows.

This should not be exposed on an untrusted network.

## Request Processing Flow

`POST /generate` currently works like this:

1. Read the request body as the prompt.
2. Encode the prompt into token ids.
3. Reset the in-memory KV-cache for that request.
4. Run one prefill inference over the full prompt.
5. Select the next token from the final logits window.
6. Re-run inference one token at a time while reusing KV-cache state.
7. Decode generated token ids and return JSON metadata.

## KV-Cache Behavior

- KV-cache state is held inside the shared `ModelInference` instance.
- Cache state is reset at the start of each `/generate` request.
- Cache state is reused only within the lifetime of a single generation request.
- The response exposes cache metadata for visibility during testing and debugging.

## Operational Notes

- The application expects model artifacts to exist before startup; it does not download dependencies dynamically.
- The ONNX Runtime library path is configured in CMake via `ONNXRUNTIME_LIB`.
- The executable build uses `BUILD_RPATH` to find the ONNX Runtime shared library from the configured models directory.
- The HTTP layer currently returns plain text for error responses and JSON only for `/generate`.

## Known Limitations

- The tokenizer is byte-based and simplified. It keeps token ids inside safe ONNX bounds for this project, but it is not a true model tokenizer.
- The generation strategy is greedy argmax only.
- There is no streaming response mode.
- There is no authentication, rate limiting, structured logging, or configuration layer.
- The service is tuned for local development, not multi-tenant or internet-facing deployment.

## Common Failure Modes

### Build fails with missing ONNX Runtime library

Cause:

- `models/libonnxruntime.so` does not exist
- `ONNXRUNTIME_LIB` points to a missing file

Fix:

- Download ONNX Runtime for your platform
- Extract the archive
- Point `models/libonnxruntime.so` and `models/libonnxruntime.so.1` at the real library file

### Runtime fails with `libonnxruntime.so.1` not found

Cause:

- The SONAME symlink is missing

Fix:

- Ensure both of these exist:
  - `models/libonnxruntime.so`
  - `models/libonnxruntime.so.1`

### `POST /generate` returns an error

Cause:

- `models/model.onnx` is missing
- The model export does not match the inference input/output assumptions
- The prompt is empty

Fix:

- Place the ONNX model at `models/model.onnx`
- Confirm the model expects the current `input_ids`, `attention_mask`, `position_ids`, and KV-cache inputs
- Send a non-empty prompt body

## Development Notes

If you change the model contract, review these areas together:

- [routes.cpp](/home/jake/cpp-projects/fictional-funicular/src/http/routes.cpp)
- [model_inference.cpp](/home/jake/cpp-projects/fictional-funicular/src/inference/model_inference.cpp)
- [tokenizer.cpp](/home/jake/cpp-projects/fictional-funicular/src/tokenizer/tokenizer.cpp)
- [inference_tests.cpp](/home/jake/cpp-projects/fictional-funicular/tests/unit/inference_tests.cpp)
- [routes_tests.cpp](/home/jake/cpp-projects/fictional-funicular/tests/unit/routes_tests.cpp)
