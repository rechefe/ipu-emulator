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
    src/lib/ipu/ipu.c
    src/lib/ipu/ipu.h
)

# Build xmem library
add_library(xmem
    src/lib/xmem/xmem.c
    src/lib/xmem/xmem.h
)

# Logger library
add_library(logger
    src/lib/logging/logger.c
    src/lib/logging/logger.h
)

include_directories(src/lib)

# Build the app
add_executable(fully_connected
    src/apps/fully_connected.c
)

# Link the app to both libraries
target_link_libraries(fully_connected
    ipu
    xmem
    logger
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
target_link_libraries(ipu_tests PRIVATE ipu xmem logger)

# Ensure tests have access to logger too
target_link_libraries(xmem_tests PRIVATE logger)

## Helper macro to add small tools/executables that reuse libraries
macro(add_tool name)
    add_executable(${name} ${ARGN})
    # link common libs by default; callers can add more
    target_link_libraries(${name} PRIVATE ipu xmem logger)
    add_custom_target(${name}_run
        COMMAND ${name}
        DEPENDS ${name}
        COMMENT "Run ${name}" VERBATIM
    )
endmacro()

# Add tool stubs
add_tool(ipu-emulator src/tools/ipu-emulator.c)
add_tool(ipu-assembler src/tools/ipu-assembler.c)

# Meta target for building all tools
add_custom_target(tools DEPENDS ipu-emulator ipu-assembler)
