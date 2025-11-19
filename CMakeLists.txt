cmake_minimum_required(VERSION 3.16)
project(MyProject C)

set(CMAKE_C_STANDARD 11)

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