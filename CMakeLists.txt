cmake_minimum_required(VERSION 3.16)
project(MyProject C CXX)

set(CMAKE_C_STANDARD 11)

# Recommended warning flags for GCC/Clang to surface issues
if(NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic)
else()
    add_compile_options(/W4)
endif()

# Option to enable coverage instrumentation
option(ENABLE_COVERAGE "Enable coverage flags (uses lcov/genhtml)" OFF)
if(ENABLE_COVERAGE)
    message(STATUS "Coverage instrumentation enabled")
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -O0 -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

# Download and make GoogleTest available for building tests
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Add include directories (if using include/)
include_directories(include)

# Build ipu library
add_library(ipu
    src/ipu/ipu.c
    src/ipu/ipu.h
)

# Build xmem library
add_library(xmem
    src/xmem/xmem.c
    src/xmem/xmem.h
)

include_directories(src)

# Build the app
add_executable(fully_connected
    src/apps/fully_connected.c
)

# Link the app to both libraries
target_link_libraries(fully_connected
    ipu
    xmem
)

# xmem unit tests
## Helper macro to simplify adding C tests that use GoogleTest
macro(add_c_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE gtest_main)
    add_test(NAME ${target} COMMAND ${target})
endmacro()

# Add xmem tests
add_c_test(xmem_tests tests/test_xmem.cpp)
target_link_libraries(xmem_tests PRIVATE xmem)

# Add ipu tests (see tests/test_ipu.cpp)
add_c_test(ipu_tests tests/test_ipu.cpp)
target_link_libraries(ipu_tests PRIVATE ipu xmem)