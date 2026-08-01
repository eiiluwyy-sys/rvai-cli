# RISC-V cross-build and QEMU preparation

P3.1 proves that the existing native workload can be cross-compiled into a
real 64-bit RISC-V Linux executable. It does not run the binary through QEMU
and does not change the Python CLI.

## Prerequisites

On Ubuntu, install the GNU riscv64 Linux cross-toolchain:

```bash
sudo apt-get update
sudo apt-get install g++-riscv64-linux-gnu binutils-riscv64-linux-gnu
```

No RVV compiler options are enabled in this phase. The generated code uses the
cross-compiler's default RISC-V ISA and ABI.

## Build

Use the repository script:

```bash
./scripts/build-riscv64.sh
```

It is equivalent to:

```bash
cmake \
  -S . \
  -B build-riscv64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-linux-gnu.cmake

cmake --build build-riscv64 --parallel
```

The native and cross-compiled build trees remain separate:

```text
build/          native host build
build-riscv64/  riscv64 Linux cross-build
```

Both builds compile the same `native/rvai_bench.cpp` source file.

## Verify the ELF file

```bash
file build-riscv64/rvai-bench
riscv64-linux-gnu-readelf -h build-riscv64/rvai-bench
riscv64-linux-gnu-readelf -A build-riscv64/rvai-bench
```

The output must include:

```text
ELF 64-bit
Machine:                           RISC-V
```

`readelf -A` displays the RISC-V attributes emitted by the selected toolchain;
older toolchains may not emit an attributes section. P3.1 does not require,
enable, or claim RVV support.

## QEMU scope

Executing `rvai-bench` with QEMU user-mode emulation is intentionally deferred
to a later phase. This phase only establishes the reproducible cross-build and
ELF inspection boundary that QEMU execution will consume.
