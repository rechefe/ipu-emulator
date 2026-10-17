"""Benchmark modules: each one here is found and run by ``ipu_bench.bench``.

A module exposes ``run(budget: Budget) -> list[Result]`` and takes each
measurement in a fresh interpreter (:func:`ipu_bench.isolated.call`). Adding a
module adds its results; the runner does not change.
"""
