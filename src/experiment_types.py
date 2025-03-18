from dataclasses import dataclass
from typing import Union

from model_types import ModelConfig, TrainingMetrics


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment with fixed parameters."""

    cores: int
    executor_memory_size: str  # e.g. "2g"
    driver_memory_size: str  # e.g. "4g"
    instances: int
    model_config: ModelConfig
    test_size: float
    train_size: Union[float, None] = None

    def as_pretty_dict(self) -> dict[str, Union[str, int]]:
        """Return config as a dictionary with formatted values."""
        result: dict[str, Union[str, int]] = {
            "cores": self.cores,
            "executor_memory_size": int(float(self.executor_memory_size[:-1]) * 1000),
            "driver_memory_size": int(float(self.driver_memory_size[:-1]) * 1000),
            "instances": self.instances,
        }

        for key, value in self.model_config.as_pretty_dict().items():
            result[key] = value

        return result

    def as_pretty_string(self) -> str:
        """Return config as a concise one-line string."""
        return (
            f"cores={self.cores}, exmem={self.executor_memory_size}, "
            f"drmem={self.driver_memory_size}, inst={self.instances}, "
            f"{self.model_config.as_pretty_string()}"
        )


@dataclass
class ExperimentSeriesConfig:
    """Configuration for a series of scalability experiments."""

    cores: list[int]
    executor_memory_sizes: list[str]  # e.g. ["2g", "4g", "8g"]
    driver_memory_sizes: list[str]  # e.g. ["2g", "4g", "8g"]
    instances: list[int]
    model_configs: list[ModelConfig]
    test_size: float = 0.2
    train_size: Union[float, None] = None

    def as_pretty_dict(self) -> dict[str, str]:
        """Return config as a dictionary with formatted values."""
        return {
            "cores": str(self.cores),
            "memory_sizes": str(self.executor_memory_sizes),
            "driver_memory_sizes": str(self.driver_memory_sizes),
            "instances": str(self.instances),
            "model_configs": f"{len(self.model_configs)} configs",
        }

    def as_pretty_string(self) -> str:
        """Return config as a concise one-line string."""
        return (
            f"cores={self.cores}, mem={self.executor_memory_sizes}, "
            f"disk={self.driver_memory_sizes}, inst={self.instances}, "
            f"models={len(self.model_configs)}"
        )

    def generate_experiment_configs(self) -> list[ExperimentConfig]:
        """Generate all possible experiment configurations from this series."""
        configs = []
        for core in self.cores:
            for mem in self.executor_memory_sizes:
                for disk in self.driver_memory_sizes:
                    for instance in self.instances:
                        for model in self.model_configs:
                            configs.append(
                                ExperimentConfig(
                                    cores=core,
                                    executor_memory_size=mem,
                                    driver_memory_size=disk,
                                    instances=instance,
                                    model_config=model,
                                    test_size=self.test_size,
                                    train_size=self.train_size,
                                )
                            )
        return configs


@dataclass
class ExperimentResult:
    """Results from a scalability experiment."""

    config: ExperimentConfig  # Experiment configuration
    metrics: TrainingMetrics  # Training metrics
    resource_metrics: Union[dict, None] = None  # Resource utilization metrics

    def as_pretty_dict(self) -> dict[str, Union[str, int]]:
        """Return results as a dictionary with formatted values."""
        # merge config, metrics, and resource_metrics into a single dictionary
        result: dict[str, Union[str, int]] = {}

        for key, value in self.config.as_pretty_dict().items():
            result[key] = value
        for key, value in self.metrics.as_pretty_dict().items():
            result[key] = value
        if self.resource_metrics:
            for key, value in self.resource_metrics.items():
                result[key] = value

        return result

    def as_pretty_string(self) -> str:
        """Return results as a concise one-line string."""
        # Summarize resource metrics with key stats if available
        resource_summary = ""
        if self.resource_metrics is not None:
            key_metrics = ["cpu_util", "mem_util", "disk_util"]
            resource_summary = ", ".join(
                f"{k}={self.resource_metrics.get(k, 'N/A'):.4g}"
                for k in key_metrics
                if k in self.resource_metrics and isinstance(self.resource_metrics[k], (int, float))
            )
            if resource_summary:
                resource_summary = f", {resource_summary}"

        return f"{self.metrics.as_pretty_string()}{resource_summary}, {self.config.as_pretty_string()}"
