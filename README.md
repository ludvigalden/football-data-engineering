# Scalable Football Match Data Analytics

This project implements a scalable data processing pipeline for football match data analysis using Apache Spark. It demonstrates data engineering for handling medium-sized datasets with an emphasis on transformation efficiency, feature engineering, and horizontal scalability.

## Background

Football (soccer) stands as the world's most popular sport. It generates enormous amounts of data through matches, player statistics, and betting markets. This data explosion has transformed how teams, analysts, and fans understand the game, moving from subjective observations to data-driven decision-making. The quantitative analysis of football data began in earnest during the early 2000s when clubs like Liverpool FC and Arsenal started employing data analysts. This trend accelerated following the success of "Moneyball" approaches in baseball, with European football clubs increasingly adopting sophisticated data analytics to gain competitive advantages.

The football analytics landscape has evolved significantly over the past decade. Initially focused on basic statistics like possession percentages and shot counts, modern analysis now incorporates complex metrics such as expected goals (xG), pressing intensity, and progressive passes. Companies like StatsBomb, Opta, and Wyscout have emerged as major data providers, while betting companies utilize vast datasets to set increasingly accurate odds.

While previous research has explored various aspects of football prediction using similar datasets, these efforts have often been limited by computational constraints when handling large-scale historical data. For example, Hubáček et al. (2019) achieved 52.5% accuracy in predicting match outcomes using ensemble methods, but their approach was limited to a single league and struggled with scaling to multiple competitions simultaneously.

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

## Computational Experiments

Our computational experiments demonstrate a data processing pipeline using Apache Spark, focusing on horizontal scalability, data transformations, and resource utilization analysis. The architecture consists of data ingestion, transformation, feature engineering, and analysis layers, though our implementation is relatively straightforward compared to enterprise-scale systems.

We implemented a basic prediction system for football match outcomes using Spark's DataFrame API and MLlib. The pipeline converts CSV data to Parquet format, achieving approximately 70% storage reduction and a 66x speedup in read performance. This dramatic improvement underscores the importance of appropriate data format selection for analytical workloads, as Parquet's columnar storage with compression significantly outperforms row-based CSV files.

Our partitioning strategy organizes data by country and league, creating 19 country partitions and 29 league partitions. This enables more efficient filtering when querying specific competitions, though the overall dataset size (approximately 45MB) means partitioning benefits are modest compared to what would be seen with truly large-scale data.

For our scalability tests, we systematically varied executor memory (1g-16g), driver memory (1g-16g), and core count (1-8) to identify how resource allocation affects processing performance. We implemented and compared two prediction approaches: logistic regression and gradient-boosted trees (GBT). Our experiments reveal several interesting patterns in how these resources affect training time, as shown in the plots below:

![Training time across different resource configurations](plots/training_time.png)

The first plot shows training time versus executor memory size, the second shows training time versus driver memory size, and the third shows training time versus number of cores. Each plot displays results for both logistic regression (blue) and GBT (red) models.

We also measured prediction accuracy across different resource configurations, though as expected, changing computational resources did not affect model accuracy:

![Accuracy across different resource configurations](plots/accuracy.png)

## Analysis

Our experiments reveal several unexpected insights about the scalability characteristics of our Spark-based football data processing pipeline. Most notably, we observed that simply increasing computational resources did not consistently improve performance—in fact, it often degraded it.

For both logistic regression and GBT models, we found that increasing memory allocation from 1g to 2g yielded performance improvements, but further increases showed diminishing or even negative returns. In the mid-range memory allocations (4g-10g), performance fluctuated unpredictably, while larger allocations (10g-16g) sometimes led to longer processing times. This counter-intuitive behavior likely stems from increased garbage collection overhead and memory management complexity when excessive resources are allocated to a relatively small dataset.

Even more surprising were our findings regarding core count scaling. Performance improved when scaling from 1 to 2 cores, but beyond that, processing time generally increased with additional cores. With 8 cores, training time was actually longer than with a single core. This clearly demonstrates that for our dataset—which is relatively small at 45MB—the overhead of task distribution and coordination outweighs the benefits of parallel processing beyond a certain point. This is a textbook example of the "small data problem" where distributed computing introduces more overhead than benefit.

The choice of machine learning algorithm significantly impacted processing time, with GBT models consistently requiring approximately 4.5 times longer training time than logistic regression across all resource configurations. This difference remained consistent regardless of the resource allocation, highlighting that algorithm complexity is a major factor in computational requirements.

Despite controlled experimental conditions, we observed non-deterministic performance patterns, particularly in mid-range resource configurations. This highlights the complexity of Spark's resource management and task scheduling mechanisms, which involve numerous internal optimizations and trade-offs that aren't always transparent to the user.

## Implications

The findings from our computational experiments have important implications for designing data engineering pipelines, particularly when working with small to medium-sized datasets. Most significantly, our results challenge the common assumption that "more is better" when it comes to computational resources. For our dataset, the optimal configuration was modest: 2 cores with 2-4g of memory for both executor and driver. This right-sizing of resources is crucial not only for performance but also for cost-efficiency in cloud environments where resources are billed by usage.

The dramatic performance improvement achieved by converting from CSV to Parquet format (66x speedup) reinforces the critical importance of storage format selection in data engineering pipelines. While this is a well-known best practice, seeing such a significant improvement with a relatively small dataset emphasizes that format optimization should be one of the first considerations in any data pipeline design, regardless of scale.

Our experience with diminishing or negative returns when scaling beyond certain resource thresholds highlights an important consideration for distributed computing: there exists a "sweet spot" where the benefits of parallelization are maximized relative to the overhead it introduces. For small to medium-sized datasets like ours, highly distributed processing may actually be counterproductive. This suggests that data engineers should consider data size thresholds when deciding between single-node and distributed processing approaches.

The consistent performance difference between logistic regression and GBT models reminds us that algorithm selection has significant implications for computational requirements. In production environments where real-time or near-real-time processing is needed, simpler algorithms may be preferable even if they offer marginally lower accuracy.

While our experiments were conducted on a relatively small dataset, the insights gained are valuable for understanding how Spark behaves under different configurations. In real-world scenarios with truly large datasets (terabytes or petabytes), we would expect different scaling patterns, likely with more consistent benefits from increased parallelization and memory allocation. However, the principle of finding the right balance between resources and workload remains universally applicable.

These findings demonstrate that effective data engineering requires not just technical implementation skills, but also a nuanced understanding of how distributed systems behave under different workloads and resource configurations. By systematically testing and measuring performance across various configurations, data engineers can optimize both performance and resource utilization, leading to more efficient and cost-effective data processing pipelines.

## References

Hubáček, Ondřej & Šír, Gustav & Železný, Filip. (2019). *Exploiting sports-betting market using machine learning.* International Journal of Forecasting. 35. doi: 10.1016/j.ijforecast.2019.01.001.
