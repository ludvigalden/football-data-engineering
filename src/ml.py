"""Simple prediction model for football match outcomes."""

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

from utils import time_execution


@time_execution
def prepare_features(df: DataFrame):
    feature_cols = ["HomeElo", "AwayElo", "EloDiff", "Form3Home", "Form3Away"]

    for col in feature_cols:
        df = df.na.fill(0, [col])

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="Features")
    indexer = StringIndexer(inputCol="FTResult", outputCol="Label")

    pipeline = Pipeline(stages=[assembler, indexer])

    return pipeline.fit(df).transform(df)


@time_execution
def train_and_evaluate(df: DataFrame, test_size: float) -> tuple[LogisticRegressionModel, DataFrame, float]:
    prepared_df, _ = prepare_features(df)

    # Split into training and test sets
    train_df, test_df = prepared_df.randomSplit([1.0 - test_size, test_size], seed=42)

    # Train logistic regression model
    lr = LogisticRegression(featuresCol="Features", predictionCol="Prediction", labelCol="Label", maxIter=10)
    model = lr.fit(train_df)

    # Make predictions
    predictions = model.transform(test_df)

    # Evaluate model
    evaluator = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="Prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    return model, predictions, accuracy
