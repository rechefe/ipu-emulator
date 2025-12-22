ipu-c-samples
===============

[![CI](https://github.com/rechefe/ipu-emulator/actions/workflows/ci.yml/badge.svg)](https://github.com/rechefe/ipu-emulator/actions/workflows/ci.yml)
[![Documentation](https://github.com/rechefe/ipu-emulator/actions/workflows/docs.yml/badge.svg)](https://github.com/rechefe/ipu-emulator/actions/workflows/docs.yml)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://rechefe.github.io/ipu-emulator/)

Small C project that emulates simple IPU and XMEM behavior. Includes a sample `fully_connected` app and unit tests using GoogleTest.

Build
-----
This project uses Bazel for building and managing dependencies.

From the repository root:

```bash
bazel build //...
```

Run the sample app:

```bash
bazel run //:fully_connected
```

Run tests
---------
To run all tests:

```bash
bazel test //...
```

To run specific tests:

```bash
bazel test //:xmem_tests
bazel test //:ipu_tests
```

To run tests with verbose output:

```bash
bazel test //... --test_output=all
```

Coverage
--------
To generate coverage reports:

```bash
bazel coverage //...
```

The coverage report will be generated at `bazel-out/_coverage/_coverage_report.dat`. To view it in lcov format:

```bash
genhtml bazel-out/_coverage/_coverage_report.dat --output-directory coverage
# open coverage/index.html in your browser
```

Adding new tests
----------------
To add a new test, add a `cc_test` target in the appropriate `BUILD` file:

```python
cc_test(
    name = "my_tests",
    srcs = ["test/test_my_module.cpp"],
    deps = [
        ":xmem",
        ":ipu",
        "@com_google_googletest//:gtest_main",
    ],
)
```

Notes
-----
- Bazel automatically handles dependency fetching (GoogleTest, cxxopts).
- Project headers are in `src/lib/` and included with paths like `#include "xmem/xmem.h"` or `#include "ipu/ipu.h"`.
- Bazel uses hermetic builds with automatic caching and parallelization.

CI
--
A GitHub Actions workflow has been added at `.github/workflows/ci.yml`. It configures the project with coverage enabled, runs the tests, captures coverage with `lcov`, generates an HTML report with `genhtml`, and uploads the coverage report as a workflow artifact.

Coverage badge
--------------
The project uploads coverage to Codecov in CI. Once the workflow runs on GitHub, the coverage badge below will reflect the latest commit coverage.

[![codecov](https://codecov.io/gh/rechefe/ipu-emulator/branch/master/graph/badge.svg)](https://codecov.io/gh/rechefe/ipu-emulator)
