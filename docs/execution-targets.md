# Execution targets

`rvai run` supports a reusable execution-target layer between the workload
adapter and the process launcher:

```text
CLI
  -> BuiltinAdapter
      -> NativeTarget
      -> QemuRiscv64Target
```

The CLI selects a target, the adapter defines GEMM arguments and validates the
result, and the target owns executable discovery plus command construction.

## Native target

The default and explicit native commands are equivalent:

```bash
rvai run builtin-gemm-int8
rvai run builtin-gemm-int8 --target native
```

`NativeTarget` launches `build/rvai-bench` directly. Set `RVAI_BENCH_BIN` when
the native executable is stored elsewhere.

## QEMU riscv64 target

Build the RISC-V executable first, then run:

```bash
./scripts/build-riscv64.sh
rvai run builtin-gemm-int8 --target qemu-riscv64
```

The target supports these overrides:

```text
RVAI_QEMU_RISCV64_BIN   qemu-riscv64 executable
RVAI_RISCV64_SYSROOT    target Linux sysroot
RVAI_RISCV64_BENCH_BIN  riscv64 rvai-bench executable
```

Defaults are `qemu-riscv64`, `/usr/riscv64-linux-gnu`, and
`build-riscv64/rvai-bench`. The target passes launcher-declared QEMU metadata
to the guest and the adapter rejects results that do not report riscv64,
qemu-user, and non-representative performance.

Automated correctness jobs can reduce the builtin GEMM workload without
changing the public CLI by setting `RVAI_GEMM_M`, `RVAI_GEMM_N`,
`RVAI_GEMM_K`, and `RVAI_GEMM_ITERATIONS`. Each value must be a positive
integer. Normal runs keep the documented `256 x 256 x 256` matrix and 20 timed
iterations when these variables are unset.

## Dry-run

Dry-run records the selected target without checking binaries or sysroots:

```bash
rvai run builtin-gemm-int8 --target qemu-riscv64 --dry-run
```

This command remains usable before QEMU or the cross-compiled ELF is present.
