ipu-c-samples
===============

Small C project that emulates simple IPU and XMEM behavior. Includes a sample `fully_connected` app and unit tests using GoogleTest.

Build
-----
This project uses CMake and FetchContent to pull GoogleTest automatically.

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j 4
```

Run the sample app:

```bash
./build/fully_connected
```

Run tests
---------
To build the test binaries and run them via CTest:

```bash
cmake --build build --target xmem_tests -j 4
cmake --build build --target ipu_tests -j 4
ctest --test-dir build --verbose
```

Or build all tests at once:

```bash
cmake --build build --target all -j 4
ctest --test-dir build --verbose
```

Adding new tests
----------------
`CMakeLists.txt` provides a small macro to make adding tests simple:

- Use `add_c_test(<target_name> <sources...>)` to create a test executable. The macro links `gtest_main` and registers the test with CTest.
- Link any project libraries required by the test with `target_link_libraries(<target_name> PRIVATE <libs...>)`.

Example (already used in the project):

```cmake
add_c_test(my_tests tests/test_my_module.cpp)
target_link_libraries(my_tests PRIVATE xmem ipu)
```

Notes
-----
- Building tests requires network access the first time (FetchContent downloads GoogleTest).
- The project headers are in `src/` and are included with paths like `#include "xmem/xmem.h"` or `#include "ipu/ipu.h"`.

If you want, I can also add a GitHub Actions workflow to run the tests automatically on push/PR.

Coverage
--------
You can enable coverage instrumentation via CMake by passing `-DENABLE_COVERAGE=ON` when configuring:

```bash
cmake -S . -B build -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j 4
ctest --test-dir build --verbose
lcov --directory build --capture --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info
genhtml coverage.info --output-directory coverage-report

# open coverage-report/index.html in your browser
```

CI
--
A GitHub Actions workflow has been added at `.github/workflows/ci.yml`. It configures the project with coverage enabled, runs the tests, captures coverage with `lcov`, generates an HTML report with `genhtml`, and uploads the coverage report as a workflow artifact.

Coverage badge
--------------
The project uploads coverage to Codecov in CI. Once the workflow runs on GitHub, the coverage badge below will reflect the latest commit coverage.

[![codecov](https://codecov.io/gh/rechefe/ipu-emulator/branch/master/graph/badge.svg)](https://codecov.io/gh/rechefe/ipu-emulator)
