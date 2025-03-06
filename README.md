# Scalable Football Match Data Analytics

This project implements a scalable data processing pipeline for football match data analysis using Apache Spark. It demonstrates data engineering for handling medium-sized datasets with an emphasis on transformation efficiency, feature engineering, and horizontal scalability.

## Background

Football (soccer) stands as the world's most popular sport. It generates enormous amounts of data through matches, player statistics, and betting markets. This data explosion has transformed how teams, analysts, and fans understand the game, moving from subjective observations to data-driven decision-making. The quantitative analysis of football data began in earnest during the early 2000s when clubs like Liverpool FC and Arsenal started employing data analysts. This trend accelerated following the success of "Moneyball" approaches in baseball, with European football clubs increasingly adopting sophisticated data analytics to gain competitive advantages.

The football analytics landscape has evolved significantly over the past decade. Initially focused on basic statistics like possession percentages and shot counts, modern analysis now incorporates complex metrics such as expected goals (xG), pressing intensity, and progressive passes. Companies like StatsBomb, Opta, and Wyscout have emerged as major data providers, while betting companies utilize vast datasets to set increasingly accurate odds.

While previous research has explored various aspects of football prediction using similar datasets, these efforts have often been limited by computational constraints when handling large-scale historical data. For example, Hubáček et al. (2019) achieved 52.5% accuracy in predicting match outcomes using ensemble methods, but their approach was limited to a single league and struggled with scaling to multiple competitions simultaneously.

## Implementation Details

### Data Processing Pipeline

Our implementation follows a modular approach with clearly defined stages:

1. **Data Loading**: We load match data and ELO ratings from CSV files using Spark's DataFrame API.

2. **Data Transformation**: 
   - Convert raw data to Parquet format for better performance
   - Clean and standardize column names
   - Handle missing values through appropriate imputation strategies
   - Partition data by country and league for efficient queries

3. **Feature Engineering**:
   - Calculate team form metrics based on historical performance
   - Derive ELO rating differentials between teams
   - Generate features that capture team strength and recent performance

4. **Model Training**:
   - Prepare features for machine learning
   - Train a logistic regression model to predict match outcomes
   - Evaluate model performance across different leagues

5. **Scalability Testing**:
   - Benchmark processing time with varying computational resources
   - Analyze the relationship between cores and execution time
   - Identify bottlenecks in the processing pipeline

### Key Technical Features

- **Distributed Processing**: Leverages Spark's distributed computing capabilities
- **Efficient Storage**: Uses Parquet format with appropriate partitioning
- **Modular Design**: Separates concerns into reusable Python modules
- **Performance Optimization**: Implements caching and join optimizations
- **Scalability Analysis**: Provides quantitative measures of horizontal scaling

## Project Structure

- **/.devcontainer/** - Development container configuration for consistent environment setup

- **/data/** - Contains source data files and processed outputs
  - **/data/ELO_RATINGS.csv** - Team strength ratings captured bi-monthly
  - **/data/MATCHES.csv** - Match data including results and team performance
  - **/data/processed/** - Directory for transformed Parquet data

- **/notebooks/** - Jupyter notebooks demonstrating the data pipeline workflow
  - **/notebooks/01_data_exploration.ipynb** - Initial exploration of dataset structure and characteristics
  - **/notebooks/02_data_transformation.ipynb** - Data cleaning and conversion to Parquet format
  - **/notebooks/03_feature_engineering.ipynb** - Creation of features for match prediction
  - **/notebooks/04_model_training.ipynb** - Implementation of a logistic regression model
  - **/notebooks/05_scalability_testing.ipynb** - Performance benchmarking with varying resources

- **/src/** - Core Python modules implementing the data processing pipeline
  - **/src/config.py** - Configuration parameters for Spark and data paths
  - **/src/data_loader.py** - Utilities for loading and saving data
  - **/src/data_transformer.py** - Functions for cleaning and transforming match data
  - **/src/feature_engineering.py** - Functions for generating prediction features
  - **/src/ml.py** - Implementation of the match outcome prediction model
  - **/src/spark_session.py** - Utilities for creating and configuring Spark sessions
  - **/src/utils.py** - Helper functions for timing execution and visualization

## Development

### Using VS Code Dev Container

1. **Prerequisites**:

    - Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
    - Install [Visual Studio Code](https://code.visualstudio.com/)
    - Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

2. **Open the project**:

    - Clone the repository: `git clone https://github.com/ludvigalden/football-data-engineering.git`
    - Open the folder in VS Code
    - When prompted, click "Reopen in Container" or use the command palette (F1) and select "Remote-Containers: Reopen in Container"

3. **Working with the notebooks**:
    - The dev container comes with Jupyter extension pre-installed
    - Open any notebook from the `/notebooks` directory
    - Execute cells sequentially to follow the data pipeline

### Using GitHub Codespaces

1. Navigate to the GitHub repository
2. Click the "Code" button and select "Open with Codespaces"
3. Create a new codespace
4. The environment will be automatically configured according to the devcontainer.json

### Running Locally Without Dev Container

1. **Prerequisites**:

    - Python 3.10
    - Java 17 (required for Spark)

2. **Setup**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Running notebooks**:
    ```bash
    jupyter notebook
    ```

## Data Format

Our selected dataset, "Club Football Match Data (2000-2025)," which can be retrieved at https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025/data, represents one of the most comprehensive open-source collections of football match information available. Containing approximately 470,000 matches across 27 countries and 42 leagues, it makes it possible to analyze football performance at scale. The dataset consists of two primary CSV files:

**ELO_RATINGS.csv** (~500 teams, 2MB): Contains team strength ratings captured bi-monthly, Includes club name, country code, and Elo rating - approximately 500 teams over multiple years.

**MATCHES.csv** (~470,000 matches, 45MB): Match metadata (date, time, league, teams), eam performance metrics (Elo ratings, form), match results and betting odds.

The CSV format is widely used for its simplicity and compatibility but presents several challenges for distributed processing: Lack of native compression, no schema enforcement, inefficient for columnar analytics, and poor partitioning support. For our Spark-based solution, we'll instead convert the CSV data to Parquet format, which offers: column-oriented storage with efficient compression, schema enforcement and evolution, predicate pushdown for faster filtering, and native support for partitioning. This aligns with best practices for data engineering pipelines, allowing us to optimize storage and query performance while maintaining compatibility with our Spark-based analytics framework.

# Computational Experiments

Our computational experiments focus on demonstrating a scalable data processing pipeline using Apache Spark, with an emphasis on horizontal scalability and efficient data transformations.

## 3.1 Architecture Overview

We'll implement a data processing pipeline with four main components:

**Data Ingestion & Transformation Layer**: Load CSV data and convert to Parquet format. Partition data by league and season. Apply schema validation and data cleaning.

**Feature Engineering Layer**: Create team performance metrics (form, goal differentials). Calculate statistical features from historical match data. Generate time-based aggregations (rolling averages).

**Analysis Layer**: Implement a simple prediction model for match outcomes. Calculate prediction accuracy across different leagues

**Scalability Testing Layer**: Benchmark processing time with varying cluster sizes: Measure resource utilization across data transformations

## 3.2 Implementation Details

Our implementation leverages Spark's DataFrame API for structured data processing and SQL for analytical queries. We'll utilize Spark's built-in machine learning library (MLlib) for the prediction component. This involves partitioning strategies to optimize parallel processing, caching intermediate results to minimize redundant computations, broadcast joins for efficient processing of dimension tables, and dynamic resource allocation to adapt to workload changes

## 3.3 Scalability Experiments

To demonstrate horizontal scalability, we'll conduct the following experiments:

**Data Processing Scalability:** Measure processing time for the entire pipeline with 1, 2, and 4 executor cores. Compare memory usage across different configurations. Evaluate the impact of data partitioning on performance.

**Query Performance**: Benchmark complex analytical queries with varying data volumes. Measure the impact of caching on query latency. Compare execution plans for optimized vs. non-optimized queries.

**Resource Utilization**. Monitor CPU, memory, and I/O metrics during processing. Identify bottlenecks in the processing pipeline. Optimize resource allocation based on workload characteristics.

## References

Hubáček, Ondřej & Šír, Gustav & Železný, Filip. (2019). *Exploiting sports-betting market using machine learning.* International Journal of Forecasting. 35. doi: 10.1016/j.ijforecast.2019.01.001.
