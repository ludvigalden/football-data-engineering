"""Data loading utilities for football match data."""

from typing import Optional
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import os
from config import ELO_RATINGS_PATH, MATCHES_OUTPUT_DIR, MATCHES_PATH, SPARK_MASTER
from utils import time_execution


def get_data_path(path: str, local_path: Optional[bool] = None):
    # For local development vs. cluster deployment
    base_path = "" if (local_path is None and SPARK_MASTER == "local[*]") or local_path else "hdfs://192.168.2.251:9000"
    return os.path.join(base_path, path)


def load_elo_data(spark: SparkSession, local_path: Optional[bool] = None):
    elo_path = get_data_path(ELO_RATINGS_PATH, local_path)
    elo_df = spark.read.csv(elo_path, header=True, inferSchema=True)

    # Convert date strings to date type
    elo_df = elo_df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))

    # Use UpperCamelCase
    elo_df = elo_df.withColumnsRenamed({"club": "Club", "elo": "Elo", "date": "Date", "country": "Country"})

    return elo_df


def load_matches_data(spark: SparkSession, local_path: Optional[bool] = None):
    matches_path = get_data_path(MATCHES_PATH, local_path)
    matches_df = spark.read.csv(matches_path, header=True, inferSchema=True)

    # Convert date strings to date type
    matches_df = matches_df.withColumn("MatchDate", F.to_date(F.col("MatchDate"), "yyyy-MM-dd"))

    # Use common terms (League, Date)
    matches_df = matches_df.withColumnsRenamed({"Division": "League", "MatchDate": "Date"})

    return matches_df


@time_execution
def load_data(spark: SparkSession, local_path: Optional[bool] = None) -> tuple[DataFrame, DataFrame]:
    matches_df = load_matches_data(spark, local_path)
    elo_df = load_elo_data(spark, local_path)

    return matches_df, elo_df


@time_execution
def load_parquet_data(spark: SparkSession) -> DataFrame:
    return spark.read.parquet(get_data_path(MATCHES_OUTPUT_DIR))


@time_execution
def write_parquet_data(matches_transformed_df: DataFrame) -> None:
    matches_transformed_df.write.partitionBy("Country", "League").mode("overwrite").parquet(
        get_data_path(MATCHES_OUTPUT_DIR)
    )
