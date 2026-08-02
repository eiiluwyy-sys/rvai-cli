#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${RVAI_RISCV64_BUILD_DIR:-${project_root}/build-riscv64}"
toolchain_file="${project_root}/cmake/toolchains/riscv64-linux-gnu.cmake"

for tool in cmake file grep riscv64-linux-gnu-g++ riscv64-linux-gnu-readelf; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Error: required tool not found: ${tool}" >&2
    exit 1
  fi
done

cmake \
  -S "${project_root}" \
  -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${toolchain_file}"

cmake --build "${build_dir}" --parallel

echo
file "${build_dir}/rvai-bench"
elf_header="$(riscv64-linux-gnu-readelf -h "${build_dir}/rvai-bench")"
printf '%s\n' "${elf_header}"

if ! grep -Eq 'Class:[[:space:]]+ELF64' <<<"${elf_header}"; then
  echo "Error: rvai-bench is not an ELF64 binary" >&2
  exit 1
fi

if ! grep -Eq 'Machine:[[:space:]]+RISC-V' <<<"${elf_header}"; then
  echo "Error: rvai-bench is not a RISC-V binary" >&2
  exit 1
fi
riscv64-linux-gnu-readelf -A "${build_dir}/rvai-bench"
