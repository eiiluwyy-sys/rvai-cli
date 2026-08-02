#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace rvai {

struct ExecutionInfo {
    std::string target_architecture;
    std::string execution_environment;
    std::string host_architecture;
    bool performance_representative;
};

inline std::string compiled_target_architecture() {
#if defined(__riscv) && __riscv_xlen == 64
    return "riscv64";
#elif defined(__riscv) && __riscv_xlen == 32
    return "riscv32";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#else
    return "unknown";
#endif
}

inline std::string normalize_architecture(const std::string& architecture) {
    if (architecture == "x86_64" || architecture == "amd64") {
        return "x86_64";
    }
    if (architecture == "aarch64" || architecture == "arm64") {
        return "aarch64";
    }
    if (architecture == "riscv64" || architecture == "riscv32") {
        return architecture;
    }
    throw std::runtime_error(
        "unsupported RVAI_HOST_ARCHITECTURE: " + architecture
    );
}

inline ExecutionInfo detect_execution_info() {
    const std::string target_architecture = compiled_target_architecture();
    const char* environment_value = std::getenv("RVAI_EXECUTION_ENVIRONMENT");
    const std::string execution_environment =
        environment_value == nullptr ? "native" : environment_value;

    if (execution_environment != "native" &&
        execution_environment != "qemu-user") {
        throw std::runtime_error(
            "RVAI_EXECUTION_ENVIRONMENT must be native or qemu-user"
        );
    }

    const char* host_value = std::getenv("RVAI_HOST_ARCHITECTURE");
    if (execution_environment == "qemu-user" && host_value == nullptr) {
        throw std::runtime_error(
            "RVAI_HOST_ARCHITECTURE is required for qemu-user execution"
        );
    }

    const std::string host_architecture = host_value == nullptr
        ? target_architecture
        : normalize_architecture(host_value);
    return ExecutionInfo{
        target_architecture,
        execution_environment,
        host_architecture,
        execution_environment == "native",
    };
}

}  // namespace rvai
