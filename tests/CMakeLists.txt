include_directories(.)

if (BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    find_package(Eigen3 REQUIRED)
    include(GoogleTest)

    add_executable(test_svgd test_svgd.cc)
    target_link_libraries(test_svgd
      PRIVATE
      svgd
      Eigen3::Eigen
      GTest::Main
    )
    gtest_discover_tests(test_svgd)
endif()

if (BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
    find_package(Eigen3 REQUIRED)

    add_executable(benchmark_svgd benchmark_svgd.cc)
    target_link_libraries(benchmark_svgd
      PRIVATE
      svgd
      Eigen3::Eigen
      benchmark::benchmark
    )
endif()