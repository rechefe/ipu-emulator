#!/bin/bash
# Bazel --workspace_status_command (enabled in .bazelrc). STABLE_* keys are part
# of the action key of every target built with `stamp = 1`, so a new commit, or
# a working tree becoming dirty, regenerates them instead of reusing a cached
# copy that names the wrong commit.
#
# Dirty means tracked files differ from HEAD, as with `git describe --dirty`.
# MODULE.bazel.lock is excluded because Bazel itself rewrites it during a build.
if commit=$(git rev-parse HEAD 2>/dev/null); then
  if ! git diff --quiet HEAD -- . ':(exclude)MODULE.bazel.lock' 2>/dev/null; then
    commit="${commit}-dirty"
  fi
else
  commit=unknown
fi
echo "STABLE_GIT_COMMIT ${commit}"
