# RVAI Repository Instructions

## Authority and remote execution

- The local checkout is the authoritative source tree.
- Remote hosts are execution-only. Do not edit source files on a remote host.
- Synchronize source only with `~/bin/rvai-sync`.
- Run remote validation only with `~/bin/rvai-remote-test`.
- Use `~/bin/rvai-remote-shell` only when interactive inspection is necessary.
- Do not use SCP or arbitrary `rsync` commands.
- Do not create commits, push, merge, or tag unless the user explicitly requests it.

## Stable product boundaries

- Preserve `BenchmarkResult` and `RunRecord` schemas and behavior.
- Preserve the builtin GEMM workload and its native and riscv64-QEMU execution paths.
- Preserve the P4.3A `InferenceResult` schema and `rvai infer` behavior.
- P4.3B is an offline MobileNetV2 ONNX model-production pipeline. It is not a
  new runtime, inference API, execution target, or benchmark path.
- Qwen, GGUF, llama.cpp, CUDA provider integration, ESWIN NPU compilation,
  training, object detection, and physical RISC-V benchmarking are outside
  P4.3B.
- The verified MobileNetV2 INT8 Manifest is a later integration boundary, not
  part of the initial production-pipeline implementation.

## Dependencies, data, and artifacts

- Keep ONNX inference and model-production dependencies outside the base
  installation and load optional dependencies lazily.
- Do not commit production model binaries or dataset files.
- Automated tests and CI must use only tiny local fixtures and must not
  download public models or datasets.
- Dataset manifests may record stable sample identifiers, relative paths,
  labels, digests, provenance, and licensing metadata, but not dataset content.
- Generated models, reports, packages, and work directories must remain
  untracked.
- A production pipeline must verify source identity and dataset manifests
  before calibration or evaluation.

## Reproducibility and validation

- Treat pipeline configuration, sample order, preprocessing, source identity,
  dataset-manifest identity, dependency versions, and generated-artifact
  digests as reproducibility inputs.
- Use deterministic manifest order for calibration and evaluation.
- Record Python, ONNX, ONNX Runtime, NumPy, Pillow, provider, platform, CPU,
  and pipeline versions for production runs.
- Keep timestamps and host-specific paths separate from deterministic records.
- Never rewrite a failed production report to make it pass. Preserve the
  original result, then change thresholds or quantization settings only through
  an explicit reviewed configuration revision and a new run.
- CI validation must remain offline. Real-model and real-dataset validation may
  run only against pre-provisioned external inputs on the execution host.
- Remote results are evidence only; required source changes must be made in the
  authoritative local checkout and synchronized again.

## Change discipline

- Follow `docs/p4.3b-onnx-model-pipeline.md` for P4.3B design and acceptance
  requirements.
- Keep implementation increments reviewable and run validation appropriate to
  the changed boundary.
- A more specific `AGENTS.md` may add stricter requirements for its subtree but
  must not relax these repository-wide constraints.
