workspace(name = "ipu_emulator")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# GoogleTest
http_archive(
    name = "com_google_googletest",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip"],
    strip_prefix = "googletest-1.14.0",
    sha256 = "1f357c27ca988c3f7c6b4bf68a9395005ac6761f034046e9dde0896e3aba00e4",
)

# cxxopts
http_archive(
    name = "cxxopts",
    urls = ["https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.0.0.tar.gz"],
    strip_prefix = "cxxopts-3.0.0",
    sha256 = "36f41fa2a46b3c1466613b63f3fa73dc24d912bc90d667147f1e43215a8c6d00",
    build_file = "@//third_party:cxxopts.BUILD",
)
