cmake_minimum_required(VERSION 3.2)
project(03_rsvd_team_3)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/src)

find_package(Eigen3 CONFIG REQUIRED)
add_executable(main src/main.cpp
        src/RandomizedSVD.h
)
target_link_libraries(main PRIVATE Eigen3::Eigen)


find_package(Catch2 REQUIRED)
add_executable(UnitTests
        tests/test_rsvd.cpp)
target_link_libraries(UnitTests PRIVATE Catch2::Catch2WithMain)

add_executable(BenchmarkTests benchmarks/benchmark_main.cpp
)
target_link_libraries(BenchmarkTests PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(UnitTests)