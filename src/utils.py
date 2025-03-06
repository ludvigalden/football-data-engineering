"""Utility functions for the project."""

import time
from typing import TypeVar, Callable, Any, Tuple
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


T = TypeVar("T")


def time_execution(func: Callable[..., T]) -> Callable[..., Tuple[T, float]]:
    """Decorator to measure execution time of a function."""

    def wrapper(*args: Any, **kwargs: Any) -> Tuple[T, float]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {execution_time:.2f} seconds")
        return result, execution_time

    return wrapper


def plot_scalability_results(cores, times, title="Scalability Test Results"):
    """Plot scalability test results."""
    plt.figure(figsize=(10, 6))
    plt.plot(cores, times, marker="o", linestyle="-", linewidth=2)
    plt.title(title)
    plt.xlabel("Number of Cores")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)
    plt.xticks(cores)
    return plt


def count_nulls(df: DataFrame):
    return df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
