# QEMU user-mode validation

P3.2 executes the cross-compiled `rvai-bench` with real RISC-V instruction
translation under QEMU user-mode. The trusted launcher declares the execution
environment explicitly; the benchmark does not claim to detect QEMU by itself.
This metadata is not cryptographic attestation of the execution environment.

## Prerequisites

Install QEMU user-mode support in addition to the P3.1 cross-toolchain:

```bash
sudo apt-get install qemu-user
```

Build the riscv64 binary first:

```bash
./scripts/build-riscv64.sh
```

The recorded P3.2 validation used:

```text
riscv64-linux-gnu-g++ 7.5.0
qemu-riscv64 8.2.2
host architecture: x86_64
```

The cross-build and ELF inspection commands are documented in
[RISC-V cross-build and QEMU preparation](riscv64-qemu.md).

## Small correctness run

```bash
qemu-riscv64 \
  -L /usr/riscv64-linux-gnu \
  -E RVAI_EXECUTION_ENVIRONMENT=qemu-user \
  -E RVAI_HOST_ARCHITECTURE=x86_64 \
  build-riscv64/rvai-bench \
  gemm-int8 \
  --m 32 \
  --n 32 \
  --k 32 \
  --iterations 2 \
  --backend scalar
```

## Default workload run

```bash
qemu-riscv64 \
  -L /usr/riscv64-linux-gnu \
  -E RVAI_EXECUTION_ENVIRONMENT=qemu-user \
  -E RVAI_HOST_ARCHITECTURE=x86_64 \
  build-riscv64/rvai-bench \
  gemm-int8 \
  --m 256 \
  --n 256 \
  --k 256 \
  --iterations 20 \
  --backend scalar
```

Set `RVAI_HOST_ARCHITECTURE` to the real host architecture when it is not
`x86_64`. The benchmark derives `target_architecture` from compiler macros; it
does not accept a caller-provided target value.

## Required result fields

Both commands must exit with code zero and return parseable JSON containing:

```json
{
  "status": "success",
  "correctness_verified": true,
  "execution": {
    "target_architecture": "riscv64",
    "execution_environment": "qemu-user",
    "host_architecture": "x86_64",
    "performance_representative": false
  }
}
```

`RVAI_EXECUTION_ENVIRONMENT=qemu-user` also forces
`performance_representative` to `false`. The result schema rejects a QEMU
result that claims otherwise.

## Recorded validation

Small smoke workload:

```text
M=N=K=32
iterations=2
QEMU exit code=0
status=success
correctness_verified=true
target_architecture=riscv64
execution_environment=qemu-user
performance_representative=false
```

Default workload:

```text
M=N=K=256
iterations=20
QEMU exit code=0
status=success
correctness_verified=true
target_architecture=riscv64
execution_environment=qemu-user
performance_representative=false
```

## Performance warning

QEMU user-mode latency, P95, and GOPS measure an emulated execution path. They
must not be reported as RISC-V development-board performance or compared with
native board benchmark results. QEMU results validate instruction execution,
program behavior, JSON stability, and correctness only.
