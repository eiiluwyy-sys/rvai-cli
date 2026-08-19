#!/usr/bin/env bash
set -euo pipefail

readonly LLAMA_VERSION="${RVAI_LLAMA_CPP_VERSION:-b10488}"
readonly LLAMA_COMMIT="9d77fa17254e1dee4b9e92504c91611a60b1359f"
readonly LLAMA_ARCHIVE_SHA256="a006ca1a0268a3748686040d1ae021939d92ec0f58f341a30e52056298da4a0b"
readonly LLAMA_ROOT="${RVAI_LLAMA_CPP_ROOT:-/opt/rvai/llama.cpp-b10488}"
readonly LLAMA_SOURCE="${LLAMA_ROOT}/source"
readonly LLAMA_BUILD="${LLAMA_ROOT}/build"
readonly BUILD_JOBS="${RVAI_LLAMA_CPP_JOBS:-4}"
readonly SOURCE_ARCHIVE="${RVAI_LLAMA_CPP_SOURCE_ARCHIVE:-}"
readonly SPACEMIT_MODE="${RVAI_LLAMA_CPP_SPACEMIT_MODE:-auto}"

if [[ "$(uname -m)" != "riscv64" ]]; then
    echo "Error: this build script must run natively on riscv64" >&2
    exit 1
fi
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: RVAI_LLAMA_CPP_JOBS must be a positive integer" >&2
    exit 1
fi
if [[ ! "${SPACEMIT_MODE}" =~ ^(auto|on|off)$ ]]; then
    echo "Error: RVAI_LLAMA_CPP_SPACEMIT_MODE must be auto, on, or off" >&2
    exit 1
fi

compiler="${CC:-cc}"
rvv_enabled=ON
spacemit_enabled=ON
zvfh_enabled=ON
rvv_probe='#include <stddef.h>
#include <riscv_vector.h>
vint32m8_t probe(vfloat32m8_t value, size_t vl) {
    return __riscv_vfcvt_x_f_v_i32m8_rm(value, __RISCV_FRM_RNE, vl);
}'
if ! printf '%s\n' "${rvv_probe}" | "${compiler}" -x c -c -o /dev/null - \
    -march=rv64gcv -mabi=lp64d >/dev/null 2>&1; then
    if [[ "${SPACEMIT_MODE}" == "on" ]]; then
        echo "Error: ${compiler} lacks the RVV intrinsic API required by the SpacemiT backend" >&2
        exit 1
    fi
    rvv_enabled=OFF
    spacemit_enabled=OFF
    zvfh_enabled=OFF
    echo "Warning: ${compiler} lacks the required RVV intrinsic API; building the scalar CPU backend" >&2
elif [[ "${SPACEMIT_MODE}" == "off" ]]; then
    spacemit_enabled=OFF
    zvfh_enabled=OFF
elif [[ "${SPACEMIT_MODE}" == "auto" ]]; then
    zvfh_macro="$({
        printf '' | "${compiler}" \
            -march=rv64gcv_zfh_zvfh -mabi=lp64d -dM -E - 2>/dev/null || true
    } | awk '$2 == "__riscv_zvfh" { print $3; exit }')"
    if [[ -z "${zvfh_macro}" || "${zvfh_macro}" == "0" ]]; then
        spacemit_enabled=OFF
        zvfh_enabled=OFF
        echo "Warning: ${compiler} lacks usable Zvfh intrinsics; building the standard RVV CPU backend" >&2
    fi
fi

install -d -m 0755 "${LLAMA_ROOT}"
if [[ ! -e "${LLAMA_SOURCE}" ]]; then
    readonly CLONE_TEMP="$(mktemp -d "${LLAMA_ROOT}/source.tmp.XXXXXX")"
    cleanup_clone() {
        rm -rf -- "${CLONE_TEMP}"
    }
    trap cleanup_clone EXIT
    if [[ -n "${SOURCE_ARCHIVE}" ]]; then
        if [[ ! -f "${SOURCE_ARCHIVE}" ]]; then
            echo "Error: source archive does not exist: ${SOURCE_ARCHIVE}" >&2
            exit 1
        fi
        archive_sha256="$(sha256sum "${SOURCE_ARCHIVE}" | awk '{print $1}')"
        if [[ "${archive_sha256}" != "${LLAMA_ARCHIVE_SHA256}" ]]; then
            echo "Error: llama.cpp source archive SHA-256 mismatch" >&2
            exit 1
        fi
        tar -xzf "${SOURCE_ARCHIVE}" -C "${CLONE_TEMP}" \
            --no-same-owner --no-same-permissions
        if [[ ! -f "${CLONE_TEMP}/llama.cpp-b10488/CMakeLists.txt" ]]; then
            echo "Error: llama.cpp source archive layout is invalid" >&2
            exit 1
        fi
        mv -- "${CLONE_TEMP}/llama.cpp-b10488" "${LLAMA_SOURCE}"
        printf '%s\n' "${LLAMA_ARCHIVE_SHA256}" > \
            "${LLAMA_SOURCE}/.rvai-source-sha256"
    else
        git clone --depth 1 --branch "${LLAMA_VERSION}" \
            https://github.com/ggml-org/llama.cpp.git "${CLONE_TEMP}/checkout"
        actual_commit="$(git -C "${CLONE_TEMP}/checkout" rev-parse HEAD)"
        if [[ "${actual_commit}" != "${LLAMA_COMMIT}" ]]; then
            echo "Error: llama.cpp commit mismatch: ${actual_commit}" >&2
            exit 1
        fi
        mv -- "${CLONE_TEMP}/checkout" "${LLAMA_SOURCE}"
    fi
elif [[ -d "${LLAMA_SOURCE}/.git" ]]; then
    actual_commit="$(git -C "${LLAMA_SOURCE}" rev-parse HEAD)"
    if [[ "${actual_commit}" != "${LLAMA_COMMIT}" ]]; then
        echo "Error: existing llama.cpp source is not ${LLAMA_COMMIT}" >&2
        exit 1
    fi
else
    recorded_sha256="$(cat "${LLAMA_SOURCE}/.rvai-source-sha256" 2>/dev/null || true)"
    if [[ "${recorded_sha256}" != "${LLAMA_ARCHIVE_SHA256}" ]]; then
        echo "Error: existing llama.cpp archive source is not verified" >&2
        exit 1
    fi
fi

cmake -S "${LLAMA_SOURCE}" -B "${LLAMA_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_NUMBER=10488 \
    -DLLAMA_BUILD_COMMIT=9d77fa17 \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_STATIC=ON \
    -DGGML_NATIVE=ON \
    -DGGML_CPU_RISCV64_SPACEMIT="${spacemit_enabled}" \
    -DGGML_CPU_REPACK=OFF \
    -DGGML_RVV="${rvv_enabled}" \
    -DGGML_RV_ZVFH="${zvfh_enabled}" \
    -DGGML_RV_ZFH=ON \
    -DGGML_RV_ZICBOP=ON \
    -DGGML_RV_ZIHINTPAUSE=ON \
    -DGGML_RV_ZBA=ON \
    -DGGML_LLAMAFILE=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_SUBPROCESS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TOOLS=ON
cmake --build "${LLAMA_BUILD}" --parallel "${BUILD_JOBS}" --target llama-cli

readonly LLAMA_BIN="${LLAMA_BUILD}/bin/llama-cli"
if [[ ! -x "${LLAMA_BIN}" ]]; then
    echo "Error: llama-cli was not produced at ${LLAMA_BIN}" >&2
    exit 1
fi
"${LLAMA_BIN}" --version
echo "RVAI_LLAMA_CPP_RVV_ENABLED=${rvv_enabled}"
echo "RVAI_LLAMA_CPP_SPACEMIT_ENABLED=${spacemit_enabled}"
echo "RVAI_LLAMA_CPP_BIN=${LLAMA_BIN}"
