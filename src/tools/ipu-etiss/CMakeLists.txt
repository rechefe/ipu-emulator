# ETISS backend for the IPU emulator.
#
# Builds three things against an installed ETISS:
#   libIPUFuncs.so  the instruction semantics, linked into JIT-compiled blocks
#   libIPU.so       the ETISS architecture plugin (decode + CPU state)
#   ipu_etiss_run   the runner the Python driver invokes
#
# The decode tables, CPU struct and handler prototypes are generated from
# instruction_spec.py by ipu_as.gen_etiss at configure/build time, so nothing
# here duplicates ISA metadata.
#
#   cmake -S src/tools/ipu-etiss -B build/ipu-etiss \
#         -DETISS_ROOT=/path/to/etiss-install \
#         -DIPU_PYTHON=/path/to/python
#   cmake --build build/ipu-etiss --target install

cmake_minimum_required(VERSION 3.13)
project(ipu_etiss CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT ETISS_ROOT)
  message(FATAL_ERROR "Set -DETISS_ROOT to an ETISS installation prefix")
endif()
if(NOT IPU_PYTHON)
  set(IPU_PYTHON python3)
endif()

# Both the runner and the plugin link libETISS.so from the ETISS install
# tree, which is not on the default loader path.
set(CMAKE_INSTALL_RPATH "${ETISS_ROOT}/lib;${ETISS_ROOT}/lib/plugins")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen")
set(GEN_IPU_DIR "${GEN_DIR}/IPU")

# ---------------------------------------------------------------------------
# Generated sources
# ---------------------------------------------------------------------------

set(GEN_OUTPUTS
  "${GEN_IPU_DIR}/IPU_gen.h"
  "${GEN_IPU_DIR}/IPUFuncs_gen.h"
  "${GEN_IPU_DIR}/IPUDecode_gen.cpp"
  "${GEN_IPU_DIR}/ipu_etiss_layout.py"
)

file(GLOB_RECURSE GEN_INPUTS
  "${REPO_ROOT}/src/tools/ipu-common/src/ipu_common/*.py"
  "${REPO_ROOT}/src/tools/ipu-as-py/src/ipu_as/*.py"
)

add_custom_command(
  OUTPUT ${GEN_OUTPUTS}
  COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${REPO_ROOT}/src/tools/ipu-common/src:${REPO_ROOT}/src/tools/ipu-as-py/src"
          ${IPU_PYTHON} -m ipu_as.gen_etiss --output-dir "${GEN_IPU_DIR}"
  DEPENDS ${GEN_INPUTS}
  COMMENT "Generating ETISS IPU sources from instruction_spec.py"
  VERBATIM
)
add_custom_target(ipu_etiss_gen DEPENDS ${GEN_OUTPUTS})

# ---------------------------------------------------------------------------
# libIPUFuncs.so — instruction semantics, linked into JIT blocks
# ---------------------------------------------------------------------------

add_library(IPUFuncs SHARED
  arch/IPUFuncs.c
  runtime/ipu_math.c
  runtime/ipu_activations.c
)
add_dependencies(IPUFuncs ipu_etiss_gen)
target_include_directories(IPUFuncs PUBLIC
  "${GEN_DIR}"
  "${CMAKE_CURRENT_LIST_DIR}/arch"
  "${CMAKE_CURRENT_LIST_DIR}/runtime"
  "${ETISS_ROOT}/include"
  "${ETISS_ROOT}/include/jit"
  "${ETISS_ROOT}/include/third_party/betterenums"
)
set_target_properties(IPUFuncs PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
target_link_libraries(IPUFuncs PUBLIC m)

# ---------------------------------------------------------------------------
# libIPU.so — the ETISS architecture plugin
# ---------------------------------------------------------------------------

find_library(ETISS_LIBRARY NAMES ETISS PATHS "${ETISS_ROOT}/lib" REQUIRED)

add_library(IPU SHARED
  arch/IPUArch.cpp
  arch/IPUArchLib.cpp
  "${GEN_IPU_DIR}/IPUDecode_gen.cpp"
)
add_dependencies(IPU ipu_etiss_gen)
target_include_directories(IPU PUBLIC
  "${GEN_DIR}"
  "${CMAKE_CURRENT_LIST_DIR}/arch"
  "${ETISS_ROOT}/include"
  "${ETISS_ROOT}/include/jit"
  "${ETISS_ROOT}/include/third_party/betterenums"
)
target_compile_definitions(IPU PRIVATE ETISS_EXPORTS ETISS_PLUGIN_IMPORTS)
target_link_libraries(IPU PUBLIC "${ETISS_LIBRARY}")

# ---------------------------------------------------------------------------
# ipu_etiss_run — the runner
# ---------------------------------------------------------------------------

add_executable(ipu_etiss_run runner/ipu_etiss_run.cpp)
add_dependencies(ipu_etiss_run ipu_etiss_gen)
target_include_directories(ipu_etiss_run PRIVATE
  "${GEN_DIR}"
  "${ETISS_ROOT}/include"
  "${ETISS_ROOT}/include/jit"
  "${ETISS_ROOT}/include/third_party/betterenums"
)
target_link_libraries(ipu_etiss_run PRIVATE "${ETISS_LIBRARY}")

# ---------------------------------------------------------------------------
# Install into the ETISS tree
# ---------------------------------------------------------------------------
#
# The JIT compiles each block against "<ETISS_ROOT>/include/jit" and links
# libraries from "<ETISS_ROOT>/include/jit/etiss/jit", both of which ETISS puts
# on the search path unconditionally.  Putting the generated headers and
# libIPUFuncs.so there is what lets a translated block call the semantics.

install(FILES "${GEN_IPU_DIR}/IPU_gen.h" "${GEN_IPU_DIR}/IPUFuncs_gen.h"
        DESTINATION "${ETISS_ROOT}/include/jit/IPU")
install(TARGETS IPUFuncs LIBRARY DESTINATION "${ETISS_ROOT}/include/jit/etiss/jit")
install(TARGETS IPU LIBRARY DESTINATION "${ETISS_ROOT}/lib/plugins")
install(TARGETS ipu_etiss_run RUNTIME DESTINATION "${ETISS_ROOT}/bin")
install(CODE "
  file(APPEND \"${ETISS_ROOT}/lib/plugins/list.txt\" \"IPU,${ETISS_ROOT}/lib/plugins,IPU\\n\")
")
