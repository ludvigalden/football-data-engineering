"""Data loading utilities for football match data."""

from pyspark.sql import SparkSession
from config import SPARK_MASTER, APP_NAME, EXECUTOR_CORES, EXECUTOR_MEMORY, DRIVER_MEMORY, EXECUTOR_INSTANCES


def create_spark_session(force=False) -> SparkSession:
    active_session = SparkSession.getActiveSession()
    if active_session:
        if not force:
            return active_session
        else:
            active_session.stop()

    builder: SparkSession.Builder = SparkSession.builder.master(SPARK_MASTER)  # type: ignore

    builder = (
        builder.appName(APP_NAME)
        .config("spark.log.level", "ERROR")
        .config("spark.log.level", "ERROR")
        .config("spark.executor.cores", EXECUTOR_CORES)
        .config("spark.executor.memory", EXECUTOR_MEMORY)
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.network.timeout", "120s")
        .config("spark.executor.heartbeatInterval", "20s")
    )

    if SPARK_MASTER != "local[*]":  # Not needed for local mode
        builder = builder.config("spark.executor.instances", EXECUTOR_INSTANCES)
        builder = builder.config("spark.dynamicAllocation.enabled", True)
        builder = builder.config("spark.shuffle.service.enabled", True)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    return spark
