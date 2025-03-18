from pyspark.sql import SparkSession
from config import SPARK_MASTER, APP_NAME, EXECUTOR_CORES, EXECUTOR_MEMORY, DRIVER_MEMORY, EXECUTOR_INSTANCES
import gc
import time


def create_spark_session(
    force=False,
    cores=EXECUTOR_CORES,
    executor_memory=EXECUTOR_MEMORY,
    driver_memory=DRIVER_MEMORY,
    instances=EXECUTOR_INSTANCES,
) -> SparkSession:
    active_session = SparkSession.getActiveSession()
    if active_session:
        if not force:
            return active_session
        else:
            active_session.stop()
            gc.collect()
            time.sleep(2)

    builder: SparkSession.Builder = SparkSession.builder.master(SPARK_MASTER)  # type: ignore

    builder = (
        builder.appName(APP_NAME)
        .config("spark.log.level", "ERROR")
        .config("spark.executor.cores", cores)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory", driver_memory)
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "30s")
    )

    if SPARK_MASTER != "local[*]":  # Not needed for local mode
        builder = builder.config("spark.executor.instances", instances)
        builder = builder.config("spark.dynamicAllocation.enabled", True)
        builder = builder.config("spark.shuffle.service.enabled", True)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    return spark
