# Model artifact lifecycle

RVAI model Manifests may declare one downloadable artifact with an HTTP or
HTTPS URL, a plain local filename, and a required SHA-256 digest. Existing
Manifests without an `artifact` declaration remain valid and cannot be used
with `rvai pull`.

## Download and cache

Pull and verify a declared model:

```bash
rvai pull mobilenet-v2-fp32-onnx
```

The default cache root is `~/.cache/rvai/models`. Set `RVAI_CACHE_DIR` or use
`--cache-dir` to choose another root. The resulting layout is:

```text
<cache-root>/<model-name>/<artifact-filename>
<cache-root>/<model-name>/artifact.json
```

Downloads use the Python standard library and are streamed in 1 MiB chunks.
RVAI writes a temporary file in the destination directory, calculates its
SHA-256 digest and size, flushes and synchronizes it, and publishes it with an
atomic replacement only after verification succeeds. Metadata is written by
the same temporary-file and atomic-replacement pattern.

An existing valid cache returns `already-cached` without network access. An
invalid existing file is preserved unless replacement is explicitly requested:

```bash
rvai pull mobilenet-v2-fp32-onnx --force
```

A failed forced download leaves the previous file untouched. Concurrent pulls
of the same artifact are not coordinated in this version. Each pull uses a
separate temporary file; verified identical downloads may race to be the final
atomic replacement.

## Status and integrity boundary

`rvai show <model>` never accesses the network and never hashes an entire large
model. Its `artifact.cached` field means the expected local file exists.
`artifact.verified` means the local metadata matches the current validated
Manifest, including its digest. It is a lightweight cache status, not a new
byte-level attestation.

`rvai check <model>` uses the same lightweight metadata status when deciding
whether `model_file_missing` applies. Runtime and Adapter blockers remain
independent, so a downloaded ONNX model can still have `ready: false`.

`ArtifactResolver.resolve()` is the execution boundary. It recalculates the
complete file SHA-256 and size before returning a path to an Adapter. Future
Adapters should depend only on this verified path and must not implement URL,
cache, or download logic themselves.

## Security and privacy

- Artifact filenames and model names cannot contain directory traversal.
- Only HTTP and HTTPS URLs are accepted.
- A URL never determines the local destination filename.
- Server response bodies and request headers are not included in CLI errors.
- Authentication headers, resumable downloads, archives, shards, and
  multi-file models are intentionally unsupported.
- Cached model files live outside the repository by default and generated
  repository-root artifact directories are ignored by Git.

Tests use a loopback `ThreadingHTTPServer` and a tiny binary fixture. CI never
downloads a public model.
