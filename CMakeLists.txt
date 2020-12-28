cmake_minimum_required(VERSION 3.16.0)
set(CMAKE_CXX_STANDARD 17)
project(SVGD CXX)

# Options
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_TESTS "Build tests" ON)

# Core library
add_subdirectory(src)

# Unit tests and benchmarks
add_subdirectory(tests)
