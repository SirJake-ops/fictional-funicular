# LLM Inference Engine

Small C++20 HTTP server that exposes a couple of routes and can run inference through ONNX Runtime.

## Requirements

- CMake 3.20+
- A C++20 compiler
- ONNX Runtime shared library for your platform
- A compatible ONNX model file at `models/decoder_model.onnx`

## Important: ONNX Runtime Is Not Included

This repository does not ship the ONNX Runtime `.so` file.

If you clone this project from GitHub, you must download ONNX Runtime yourself and provide the shared library locally before the project will build or run.

On Linux x86_64, the expected runtime library is typically one of these files from an ONNX Runtime release archive:

- `libonnxruntime.so.1.x.y`
- `libonnxruntime.so`

Example release archive:

- `onnxruntime-linux-x64-<version>.tgz`

After extracting the archive, you need the real shared library file from its `lib/` directory.

## Project Layout Expectations

This project currently expects:

- `models/libonnxruntime.so`
- `models/libonnxruntime.so.1`
- `models/decoder_model.onnx`

The `.so` files can be real files or symlinks to your extracted ONNX Runtime download.

Example:

```bash
ln -sfn /path/to/onnxruntime-linux-x64-1.24.4/lib/libonnxruntime.so.1.24.4 models/libonnxruntime.so
ln -sfn /path/to/onnxruntime-linux-x64-1.24.4/lib/libonnxruntime.so.1.24.4 models/libonnxruntime.so.1
```

## Build

From the project root:

```bash
cmake -S . -B .
cmake --build .
```

This produces the executable:

```bash
./LLM_Inference_Engine
```

## Run

Start the server from the project root:

```bash
./LLM_Inference_Engine
```

The server is configured to listen on:

- `127.0.0.1:1234`

Available routes:

- `GET /hi`
- `GET /run_model`
- `GET /stop`

Example:

```bash
curl http://127.0.0.1:1234/hi
curl http://127.0.0.1:1234/run_model
```

## Common Failure Modes

### Linker error for `-lonnxruntime`

Cause:

- ONNX Runtime is not installed or not wired into `models/`

Fix:

- Download ONNX Runtime
- Extract it
- Point `models/libonnxruntime.so` and `models/libonnxruntime.so.1` at the real library file

### Runtime error: `libonnxruntime.so.1` not found

Cause:

- The SONAME symlink is missing

Fix:

- Make sure both symlinks exist in `models/`:
  - `libonnxruntime.so`
  - `libonnxruntime.so.1`

### `/run_model` fails

Cause:

- `models/decoder_model.onnx` is missing
- The model is incompatible with the inference code

Fix:

- Place the ONNX model file at `models/decoder_model.onnx`

## Notes

- `httplib.h` is header-only and is already vendored in `extern/http/`
- ONNX Runtime headers are vendored in `extern/onnx/`
- The shared library binary is intentionally not committed to this repository
