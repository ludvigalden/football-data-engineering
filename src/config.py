"""Configuration parameters for the project."""

# Spark configuration
# SPARK_MASTER = "spark://192.168.2.251:7077"
SPARK_MASTER = "local[*]"
APP_NAME = "Group19_FootballDataEngineering"
EXECUTOR_CORES = 2
EXECUTOR_MEMORY = "2g"
DRIVER_MEMORY = "2g"
EXECUTOR_INSTANCES = 2

# Data paths
DATA_DIR = "./data"
ELO_RATINGS_PATH = f"{DATA_DIR}/ELO_RATINGS.csv"
MATCHES_PATH = f"{DATA_DIR}/MATCHES.csv"
OUTPUT_DIR = f"{DATA_DIR}/processed"
MATCHES_OUTPUT_DIR = f"{OUTPUT_DIR}/matches"
