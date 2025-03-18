"""Simple prediction model for football match outcomes."""

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql import DataFrame

from model_types import ModelConfig
from experiment_types import TrainingMetrics
from utils import time_execution
import time


@time_execution
def prepare_features(df: DataFrame, feature_cols=None):
    """Prepare features for training."""

    if feature_cols is None:
        feature_cols = [
            "HomeElo",
            "AwayElo",
            "EloDiff",
            "Form3Home",
            "Form3Away",
            "OddHome",
            "OddDraw",
            "OddAway",
            "Over25",
            "Under25",
        ]

    for col in feature_cols:
        df = df.na.fill(0, [col])

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="Features")
    indexer = StringIndexer(inputCol="FTResult", outputCol="Label")

    pipeline = Pipeline(stages=[assembler, indexer])
    return pipeline.fit(df).transform(df)


def train_and_evaluate(
    df: DataFrame, config: ModelConfig, test_size: float = 0.2, train_size: float | None = None
) -> tuple[TrainingMetrics, DataFrame]:
    """Train model and return metrics + predictions."""

    prepared_df, prep_time = prepare_features(df)

    train_df, test_df, _ = prepared_df.randomSplit(
        [
            1.0 - test_size if train_size is None else train_size,
            test_size,
            0 if train_size is None else 1 - test_size - train_size,
        ],
        seed=42,
    )

    # Train model
    model = config.create_model()
    train_start = time.time()
    fitted_model = model.fit(train_df)
    train_time = time.time() - train_start

    # Make predictions
    pred_start = time.time()
    predictions = fitted_model.transform(test_df)
    pred_time = time.time() - pred_start

    # Evaluate
    evaluator = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="Prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    metrics = TrainingMetrics(
        accuracy=accuracy,
        training_time=train_time,
        prediction_time=pred_time,
        total_time=prep_time + train_time + pred_time,
    )

    return metrics, predictions
