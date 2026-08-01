#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
    int m = 256;
    int n = 256;
    int k = 256;
    int iterations = 20;
    std::string backend = "scalar";
};

int parse_positive_int(const std::string& raw, const std::string& option) {
    std::size_t consumed = 0;
    long long value = 0;
    try {
        value = std::stoll(raw, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(option + " must be a positive integer");
    }
    if (consumed != raw.size() || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(option + " must be a positive integer");
    }
    return static_cast<int>(value);
}

Config parse_args(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "gemm-int8") {
        throw std::runtime_error(
            "usage: rvai-bench gemm-int8 [--m N] [--n N] [--k N] "
            "[--iterations N] [--backend scalar]"
        );
    }

    Config config;
    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + option);
        }
        const std::string value = argv[++index];
        if (option == "--m") {
            config.m = parse_positive_int(value, option);
        } else if (option == "--n") {
            config.n = parse_positive_int(value, option);
        } else if (option == "--k") {
            config.k = parse_positive_int(value, option);
        } else if (option == "--iterations") {
            config.iterations = parse_positive_int(value, option);
        } else if (option == "--backend") {
            config.backend = value;
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }

    if (config.backend != "scalar") {
        throw std::runtime_error("unsupported backend: " + config.backend);
    }
    return config;
}

std::size_t checked_elements(int rows, int columns) {
    const auto left = static_cast<std::size_t>(rows);
    const auto right = static_cast<std::size_t>(columns);
    if (left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::runtime_error("matrix dimensions are too large");
    }
    return left * right;
}

void fill_inputs(
    std::vector<std::int8_t>& lhs,
    std::vector<std::int8_t>& rhs,
    const Config& config
) {
    for (int row = 0; row < config.m; ++row) {
        for (int depth = 0; depth < config.k; ++depth) {
            lhs[static_cast<std::size_t>(row) * config.k + depth] =
                static_cast<std::int8_t>(((row * 3 + depth * 5) % 15) - 7);
        }
    }
    for (int depth = 0; depth < config.k; ++depth) {
        for (int column = 0; column < config.n; ++column) {
            rhs[static_cast<std::size_t>(depth) * config.n + column] =
                static_cast<std::int8_t>(((depth * 7 + column * 11) % 15) - 7);
        }
    }
}

void gemm_scalar(
    const std::vector<std::int8_t>& lhs,
    const std::vector<std::int8_t>& rhs,
    std::vector<std::int32_t>& output,
    const Config& config
) {
    for (int row = 0; row < config.m; ++row) {
        for (int column = 0; column < config.n; ++column) {
            std::int32_t accumulator = 0;
            for (int depth = 0; depth < config.k; ++depth) {
                const auto left = static_cast<std::int32_t>(
                    lhs[static_cast<std::size_t>(row) * config.k + depth]
                );
                const auto right = static_cast<std::int32_t>(
                    rhs[static_cast<std::size_t>(depth) * config.n + column]
                );
                accumulator += left * right;
            }
            output[static_cast<std::size_t>(row) * config.n + column] = accumulator;
        }
    }
}

bool verify_result(
    const std::vector<std::int8_t>& lhs,
    const std::vector<std::int8_t>& rhs,
    const std::vector<std::int32_t>& output,
    const Config& config
) {
    for (int row = 0; row < config.m; ++row) {
        for (int column = 0; column < config.n; ++column) {
            std::int64_t reference = 0;
            for (int depth = 0; depth < config.k; ++depth) {
                reference +=
                    static_cast<std::int64_t>(
                        lhs[static_cast<std::size_t>(row) * config.k + depth]
                    ) *
                    static_cast<std::int64_t>(
                        rhs[static_cast<std::size_t>(depth) * config.n + column]
                    );
            }
            const auto actual =
                output[static_cast<std::size_t>(row) * config.n + column];
            if (reference != actual) {
                return false;
            }
        }
    }
    return true;
}

void print_result(
    const Config& config,
    bool correctness_verified,
    double mean_ms,
    double p95_ms,
    double throughput_gops,
    std::size_t input_bytes,
    std::size_t output_bytes
) {
    std::cout << std::fixed << std::setprecision(6)
              << "{\n"
              << "  \"workload\": \"builtin-gemm-int8\",\n"
              << "  \"status\": \"success\",\n"
              << "  \"backend\": \"" << config.backend << "\",\n"
              << "  \"matrix\": {\n"
              << "    \"m\": " << config.m << ",\n"
              << "    \"n\": " << config.n << ",\n"
              << "    \"k\": " << config.k << "\n"
              << "  },\n"
              << "  \"iterations\": " << config.iterations << ",\n"
              << "  \"correctness_verified\": "
              << (correctness_verified ? "true" : "false") << ",\n"
              << "  \"latency_ms\": {\n"
              << "    \"mean\": " << mean_ms << ",\n"
              << "    \"p95\": " << p95_ms << "\n"
              << "  },\n"
              << "  \"throughput_gops\": " << throughput_gops << ",\n"
              << "  \"memory_bytes\": {\n"
              << "    \"inputs\": " << input_bytes << ",\n"
              << "    \"output\": " << output_bytes << ",\n"
              << "    \"total\": " << input_bytes + output_bytes << "\n"
              << "  }\n"
              << "}\n";
}

int run_gemm(const Config& config) {
    const auto lhs_elements = checked_elements(config.m, config.k);
    const auto rhs_elements = checked_elements(config.k, config.n);
    const auto output_elements = checked_elements(config.m, config.n);

    std::vector<std::int8_t> lhs(lhs_elements);
    std::vector<std::int8_t> rhs(rhs_elements);
    std::vector<std::int32_t> output(output_elements);
    fill_inputs(lhs, rhs, config);

    gemm_scalar(lhs, rhs, output, config);
    const bool correctness_verified = verify_result(lhs, rhs, output, config);
    if (!correctness_verified) {
        throw std::runtime_error("INT8 GEMM correctness verification failed");
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(config.iterations));
    volatile std::int64_t result_sink = 0;
    for (int iteration = 0; iteration < config.iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        gemm_scalar(lhs, rhs, output, config);
        const auto finish = std::chrono::steady_clock::now();
        result_sink += output[static_cast<std::size_t>(iteration) % output.size()];
        latencies_ms.push_back(
            std::chrono::duration<double, std::milli>(finish - start).count()
        );
    }

    const double mean_ms =
        std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
        static_cast<double>(latencies_ms.size());
    std::sort(latencies_ms.begin(), latencies_ms.end());
    const auto p95_index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(latencies_ms.size())) - 1.0
    );
    const double p95_ms = latencies_ms[p95_index];
    const double operations =
        2.0 * static_cast<double>(config.m) * static_cast<double>(config.n) *
        static_cast<double>(config.k);
    const double throughput_gops = operations / (mean_ms * 1'000'000.0);

    const std::size_t input_bytes =
        (lhs.size() + rhs.size()) * sizeof(std::int8_t);
    const std::size_t output_bytes = output.size() * sizeof(std::int32_t);
    print_result(
        config,
        correctness_verified,
        mean_ms,
        p95_ms,
        throughput_gops,
        input_bytes,
        output_bytes
    );
    static_cast<void>(result_sink);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run_gemm(parse_args(argc, argv));
    } catch (const std::exception& exc) {
        std::cerr << "rvai-bench: " << exc.what() << '\n';
        return 2;
    }
}
