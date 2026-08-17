# Changelog

All notable changes to RVAI CLI are documented in this file. The project uses
[Semantic Versioning](https://semver.org/) while its public schemas keep their
own explicit version fields.

## [Unreleased] — 0.5.0.dev0

### Added

- Apache-2.0 license text and an explicit project status document.
- Packaged built-in model Manifests for installations outside a source checkout.
- Ruff, Mypy, branch coverage, package build, and wheel smoke-test gates.
- A formal P4.3B production runner for independently labelled external data,
  immutable failure evidence, deterministic reports, and verified packaging.
- Deterministic, class-balanced numeric ImageFolder Manifest preparation with
  image decoding, per-file digests, and cross-split overlap rejection.

### Changed

- Package metadata now reads the version from `rvai.__version__` as the single
  version source.
- The registry now falls back to packaged built-in Manifests after user and
  current-directory overrides.

## [0.4.0] — 2026-08-03

- Added verified MobileNetV2 FP32 artifact management.
- Added native batch-one ONNX Runtime image classification.
- Added offline ONNX inference fixtures and regression coverage.

## [0.3.0] — 2026-08-03

- Added the riscv64 cross-build and QEMU user-mode execution target.
- Added trustworthy run records, Markdown reports, and result comparison.
- Added CI validation for direct and CLI-mediated QEMU execution.

[Unreleased]: https://github.com/eiiluwyy-sys/rvai-cli/compare/v0.4.0-onnx-inference...HEAD
[0.4.0]: https://github.com/eiiluwyy-sys/rvai-cli/releases/tag/v0.4.0-onnx-inference
[0.3.0]: https://github.com/eiiluwyy-sys/rvai-cli/releases/tag/v0.3.0-riscv-qemu
