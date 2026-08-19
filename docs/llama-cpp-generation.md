# llama.cpp and Qwen INT4 generation

RVAI provides one native, single-turn text-generation path for a verified GGUF
artifact. The initial model is the official Qwen2.5-0.5B-Instruct Q4_0 file:

- repository: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- revision: `9217f5db79a29953eb74d5343926648285ec7e67`
- filename: `qwen2.5-0.5b-instruct-q4_0.gguf`
- size: `428730208` bytes
- SHA-256: `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed`
- license: Apache-2.0

The Jupiter runtime build is pinned to llama.cpp `b10488`, commit
`9d77fa17254e1dee4b9e92504c91611a60b1359f`, and produces a CPU-only
`llama-cli`. The script compile-probes the RVV intrinsic API and enables the
upstream SpacemiT CPU path only when the selected compiler also exposes usable
Zvfh vector intrinsics. Bianbu's native GCC 13 lacks intrinsic forms required
by b10488 and reports `__riscv_zvfh=0`, so the default `auto` mode uses the
scalar CPU backend on that toolchain instead of entering an uncompilable path.

## Build on Milk-V Jupiter

From the synchronized RVAI source directory:

```bash
./scripts/build-llama-cpp-spacemit.sh
export RVAI_LLAMA_CPP_BIN=/opt/rvai/llama.cpp-b10488/build/bin/llama-cli
```

The script refuses non-riscv64 hosts, verifies the exact upstream commit, and
does not replace a checkout at another revision. `RVAI_LLAMA_CPP_ROOT` and
`RVAI_LLAMA_CPP_JOBS` may override the installation root and parallel build
count. `RVAI_LLAMA_CPP_SPACEMIT_MODE` accepts `auto` (default), `on`, or `off`;
use `on` with the upstream-recommended SpacemiT toolchain when the optimized
backend is required. In b10488 the legacy `llama-cli` target is exposed by the
server build switch, but this script builds only `llama-cli` and disables
prebuilt Web UI downloads. The build prints the effective RVV and SpacemiT
backend flags so validation evidence records whether a fallback occurred.
If the board cannot reach GitHub, provision the verified official tag archive
and set `RVAI_LLAMA_CPP_SOURCE_ARCHIVE` to its board-local path. The script
accepts only the recorded b10488 archive SHA-256.

## Download and generate

```bash
rvai pull qwen-small-int4
rvai check qwen-small-int4
rvai generate qwen-small-int4 \
  --prompt "请用三句话介绍 RISC-V" \
  --max-tokens 64 \
  --threads 4 \
  --temperature 0.2 \
  --seed 42
```

The adapter executes `llama-cli` without a shell and forces offline,
single-turn, simple-I/O operation. The GGUF-embedded chat template formats the
user and optional system prompts.

## Structured result

`rvai generate` returns JSON with:

- verified model name and runtime;
- SHA-256 and character count of the prompt, without echoing prompt text;
- effective context, token, thread, batch, temperature, and seed parameters;
- generated response text;
- total wall time and llama.cpp prompt/generation timing metrics when emitted;
- runtime build identity and executable path.

Prompt text is deliberately excluded from the structured input metadata. The
response is included because it is the workload result; users should avoid
placing secrets in prompts or requesting sensitive output.
