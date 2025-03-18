from dataclasses import dataclass
from typing import Literal, Optional, Union

from pyspark.ml.classification import LogisticRegression, GBTClassifier, OneVsRest


ModelType = Literal["logistic", "gbt"]


@dataclass
class ModelConfig:
    model_type: ModelType
    max_iter: int = 10
    parallelism: int = 1
    max_memory: str = "0.5g"  # For GBT
    max_bins: Optional[int] = None  # For GBT
    max_depth: Optional[int] = None  # For GBT
    num_trees: Optional[int] = None  # For GBT

    def create_model(self):
        if self.model_type == "logistic":
            return LogisticRegression(
                featuresCol="Features",
                predictionCol="Prediction",
                labelCol="Label",
                maxIter=self.max_iter,
                family="multinomial",
            )
        elif self.model_type == "gbt":
            gbt = GBTClassifier(
                featuresCol="Features",
                predictionCol="Prediction",
                labelCol="Label",
                maxDepth=self.max_depth or 3,
                maxBins=self.max_bins or 16,
                maxIter=self.max_iter,
                maxMemoryInMB=int(float(self.max_memory[:-1]) * 1000),
            )  # Wrap with OneVsRest for multiclass support
            return OneVsRest(
                classifier=gbt,
                parallelism=self.parallelism,
                featuresCol="Features",
                predictionCol="Prediction",
                labelCol="Label",
            )
        raise ValueError(f"Unknown model type: {self.model_type}")

    def as_pretty_dict(self) -> dict[str, Union[str, int]]:
        """Return config as a dictionary with formatted values."""
        result: dict[str, Union[str, int]] = {
            "model_type": self.model_type,
            "max_iter": self.max_iter,
            "parallelism": self.parallelism,
            "max_memory": int(float(self.max_memory[:-1]) * 1000),
        }

        if self.model_type == "gbt":
            result["max_depth"] = self.max_depth if self.max_depth else 3
            result["max_bins"] = self.max_bins if self.max_bins else 16
            # result["num_trees"] = str(self.num_trees) if self.num_trees else "N/A"

        return result

    def as_pretty_string(self) -> str:
        """Return config as a concise one-line string."""
        base = f"model={self.model_type}, iter={self.max_iter}"

        if self.model_type == "gbt":
            gbt_params = f", mem={self.max_memory}"
            if self.max_depth:
                gbt_params += f", depth={self.max_depth}"
            # if self.num_trees:
            #     gbt_params += f", trees={self.num_trees}"
            return base + gbt_params

        if self.parallelism > 1:
            base += f", parallelism={self.parallelism}"

        return base


@dataclass
class TrainingMetrics:
    """Metrics from model training."""

    accuracy: float
    training_time: float
    prediction_time: float
    total_time: float

    def as_pretty_dict(self) -> dict[str, int]:
        """Return metrics as a dictionary with formatted values."""
        return {
            "accuracy": int(self.accuracy * 100),
            "training_time": int(self.training_time * 1000),
            "prediction_time": int(self.prediction_time * 1000),
            "total_time": int(self.total_time * 1000),
        }

    def as_pretty_string(self) -> str:
        """Return metrics as a concise one-line string with 4 significant digits."""
        return (
            f"acc={self.accuracy:.4g}, time={self.total_time:.4g}s, "
            + f"train={self.training_time:.4g}s, pred={self.prediction_time:.4g}s"
        )
