"""Entry point fino do benchmark principal."""

from config import BENCHMARK_MODELS
from cv_framework.benchmarking import run_benchmark


if __name__ == "__main__":
    run_benchmark(BENCHMARK_MODELS)
