cmake_minimum_required(VERSION 3.13)

add_library(ipu_runtime_math OBJECT
  ipu_math.c
  ipu_activations.c
)

set_target_properties(ipu_runtime_math PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
)

target_include_directories(ipu_runtime_math PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(ipu_runtime_math PUBLIC m)
