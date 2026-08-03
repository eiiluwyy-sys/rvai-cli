# Native ONNX image inference

P4.3A adds a local, native CPU inference path for one FP32 ONNX image
classification model. It deliberately does not add QEMU ONNX execution,
quantization, detection, batching, or multi-input model support.

## Installation and isolation

The base CLI remains installable without any model runtime:

```bash
python -m pip install -e .
```

Install image inference explicitly:

```bash
python -m pip install -e ".[onnx]"
```

The extra contains NumPy, ONNX Runtime, and Pillow. RVAI does not import these
packages at module import time. Commands such as `rvai detect`, `rvai pull`, and
the builtin GEMM continue to start when the extra is absent. `rvai infer` loads
the packages lazily and gives the installation command when they are missing.
Runtime detection uses package metadata and module discovery; it does not load
or initialize ONNX Runtime.

## Execution path

```text
CLI
→ validated Model Manifest
→ compatibility and lightweight cache check
→ ArtifactResolver full size and SHA-256 verification
→ Manifest-driven image preprocessing
→ OnnxRuntimeAdapter / CPUExecutionProvider
→ softmax and Top-K postprocessing
→ InferenceResult 1.0 JSON
```

The Adapter consumes a verified local path. It has no URL, download, cache
layout, or artifact replacement logic.

## Manifest processing contract

The MobileNet Manifest declares RGB decoding, tensor dimensions, NCHW FP32
layout, bilinear resizing, center cropping, channel normalization, score type,
Top-K count, and label catalog. For the current MobileNet model, RVAI resizes
the shorter image side to 256, center-crops 224 × 224, converts pixels to
`[0, 1]`, and applies ImageNet mean and standard deviation values.

These values follow the ONNX Model Zoo MobileNet documentation and remain data,
not Adapter conditionals. A different compatible model can declare different
dimensions or normalization without modifying `OnnxRuntimeAdapter`.

Reference: [ONNX Model Zoo MobileNet preprocessing and output contract](https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/README.md).

## Usage

```bash
rvai pull mobilenet-v2-fp32-onnx

rvai infer mobilenet-v2-fp32-onnx \
  --input examples/images/demo.jpg
```

The result is independent of `BenchmarkResult` and `RunRecord`:

```json
{
  "schema_version": "1.0",
  "model": "mobilenet-v2-fp32-onnx",
  "status": "success",
  "runtime": "onnxruntime",
  "input": {
    "path": "examples/images/demo.jpg",
    "media_type": "image",
    "original_width": 640,
    "original_height": 480,
    "tensor_shape": [1, 3, 224, 224],
    "dtype": "float32"
  },
  "predictions": [
    {
      "index": 975,
      "label": "lakeside, lakeshore",
      "score": 0.59
    }
  ],
  "execution": {
    "execution_environment": "native",
    "provider": "CPUExecutionProvider",
    "runtime_version": "1.28.0",
    "latency_ms": 2.2
  }
}
```

Latency measures one `InferenceSession.run` call. It excludes image decoding,
preprocessing, session creation, and Top-K postprocessing. It is a single
host-dependent measurement, not a benchmark or a physical RISC-V performance
claim.

## Integrity and error boundary

`rvai infer` first uses compatibility metadata to provide a fast readiness
check. It then calls `ArtifactResolver`, which rereads the complete model and
recalculates its SHA-256. A file changed after `rvai pull` is rejected before
ONNX Runtime receives it. Runtime, image decode, tensor shape, output shape,
and non-finite score failures become concise CLI errors without Python
tracebacks.

## Offline and real-model validation

CI executes a committed 185-byte ONNX model that calculates per-channel image
means. Its input image and expected Top-K result are deterministic. The fixture
can be regenerated with `scripts/generate-tiny-onnx-fixture.py`; that maintenance
script requires the `onnx` package, which is not a product dependency.
The generator fixes ONNX IR version 8 and opset version 13 independently,
reloads the serialized model, and runs the ONNX checker. Regression tests also
create a real CPUExecutionProvider session and assert input, output, and numeric
inference contracts. The fixture file size is intentionally not a compatibility
contract because protobuf encoding can change between ONNX releases.

Manual integration also downloads the declared `mobilenetv2-12.onnx`, verifies
its 13,964,571 bytes and SHA-256, preprocesses a real 640 × 480 JPEG, and runs it
with ONNX Runtime CPUExecutionProvider. The observed Top-3 scene classes were
`lakeside`, `seashore`, and `breakwater`.

The packaged ImageNet label catalog is sourced from the Apache-2.0 ONNX Model
Zoo classification [`synset.txt`](https://github.com/onnx/models/blob/main/validated/vision/classification/synset.txt).
Both model and label provenance are declared and reviewed in the repository;
CI does not download either from the network.

## Current limitations

- native CPUExecutionProvider only
- image classification only
- one input image and batch size one
- NCHW float32 tensors only
- one model input and the first model output
- no accuracy dataset evaluation
- no QEMU, RVV, NPU, INT8, or physical-board inference
