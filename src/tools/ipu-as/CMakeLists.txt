cmake_minimum_required(VERSION 3.16)

# Allow overriding sources and libs from top-level
if(NOT DEFINED IPU_AS_SOURCES)
    set(IPU_AS_SOURCES src/ipu-assembler.cpp)
endif()

if(NOT DEFINED IPU_AS_LIBS)
    set(IPU_AS_LIBS ipu xmem logger)
endif()

add_library(ipu_as_parser STATIC lib/parser/parser.cpp lib/parser/parser.h)

add_library(ipu_as_validator STATIC lib/validator/validator.cpp lib/validator/validator.h)

add_executable(ipu-as ${IPU_AS_SOURCES})
if(DEFINED cxxopts_SOURCE_DIR)
    target_include_directories(ipu-as PRIVATE ${cxxopts_SOURCE_DIR}/include)
endif()

# Make current source dir visible to the executable and tests (for headers)
target_include_directories(ipu-as PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(ipu-as PRIVATE ${IPU_AS_LIBS} ipu_as_parser)
target_link_libraries(ipu-as PRIVATE ipu_as_validator)

# Parser unit tests (declared here so they live with the module)
add_c_test(parser_tests ${CMAKE_CURRENT_SOURCE_DIR}/test/test_parser.cpp)
target_include_directories(parser_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(parser_tests PRIVATE ipu_as_parser)

# Validator tests
add_c_test(validator_tests ${CMAKE_CURRENT_SOURCE_DIR}/test/test_validator.cpp)
target_include_directories(validator_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(validator_tests PRIVATE ipu_as_parser ipu_as_validator)

install(TARGETS ipu-as RUNTIME DESTINATION bin)
