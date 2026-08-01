cmake_minimum_required(VERSION 3.16)

project(rvai_native LANGUAGES CXX)

add_executable(rvai-bench native/rvai_bench.cpp)
target_compile_features(rvai-bench PRIVATE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(rvai-bench PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS rvai-bench RUNTIME DESTINATION bin)

enable_testing()
add_test(
  NAME rvai-bench-gemm-int8
  COMMAND rvai-bench gemm-int8 --m 8 --n 8 --k 8 --iterations 2
)
set_tests_properties(
  rvai-bench-gemm-int8
  PROPERTIES PASS_REGULAR_EXPRESSION "\"correctness_verified\": true"
)
