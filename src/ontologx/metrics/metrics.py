"""Metrics evaluation module for OntologX."""

import functools

from deepeval.models.base_model import DeepEvalBaseLLM

from ontologx.metrics.llm_metrics import GEvalGraphAlignmentMetrics
from ontologx.metrics.ontology_metrics import OntologyGraphMetrics
from ontologx.metrics.shacl_metrics import SHACLMetrics
from ontologx.store import GraphDocument


class MetricsEvaluator:
    """Evaluator for accuracy metrics."""

    @functools.cached_property
    def shacl_violations_ratio(self) -> float:
        """Calculate the ratio of SHACL violations across all pred graphs."""
        return sum(metric.violations_ratio for metric in self.__shacl_metrics) / len(self.__shacl_metrics)

    @functools.cached_property
    def precision(self) -> float:
        """Calculate precision across all pred graphs."""
        return sum(metric.precision for metric in self.__ontology_metrics) / len(self.__ontology_metrics)

    @functools.cached_property
    def recall(self) -> float:
        """Calculate recall across all pred graphs."""
        return sum(metric.recall for metric in self.__ontology_metrics) / len(self.__ontology_metrics)

    @functools.cached_property
    def f1(self) -> float:
        """Calculate F1 score across all pred graphs."""
        return sum(metric.f1 for metric in self.__ontology_metrics) / len(self.__ontology_metrics)

    @functools.cached_property
    def entity_linking_accuracy(self) -> float:
        """Calculate entity linking accuracy across all pred graphs."""
        return sum(metric.entity_linking_accuracy for metric in self.__ontology_metrics) / len(self.__ontology_metrics)

    @functools.cached_property
    def relationship_linking_accuracy(self) -> float:
        """Calculate relationship linking accuracy across all pred graphs."""
        return sum(metric.relationship_linking_accuracy for metric in self.__ontology_metrics) / len(
            self.__ontology_metrics,
        )

    @functools.cached_property
    def geval_mean(self) -> float:
        """Calculate G-Eval score across all pred graphs using the provided LLM model."""
        return self.__geval_metrics.mean

    @functools.cached_property
    def geval_mean_with_compliance(self) -> float:
        """Calculate G-Eval score across all pred graphs, multiplied by the SHACL compliance ratio."""
        return self.__geval_metrics.mean_with_compliance

    def __init__(
        self,
        y_pred: list[GraphDocument],
        y_true: list[GraphDocument],
        llm_model: DeepEvalBaseLLM,
        ontology_path: str,
        shacl_path: str,
    ):
        """Initialize the evaluator with predictions and true values.

        Args:
            y_pred (list[GraphDocument]): List of predicted GraphDocuments.
            y_true (list[GraphDocument]): List of true GraphDocuments.
            llm_model (DeepEvalBaseLLM): The LLM model used for LLM evaluation.
            ontology_path (str): Path to the ontology file.
            shacl_path (str): Path to the SHACL file.

        """
        self.__ontology_metrics = [OntologyGraphMetrics(pred, true) for pred, true in zip(y_pred, y_true, strict=True)]
        self.__shacl_metrics = [SHACLMetrics(pred, ontology_path, shacl_path) for pred in y_pred]
        self.__geval_metrics = GEvalGraphAlignmentMetrics(
            y_pred,
            [sm.compliance_ratio for sm in self.__shacl_metrics],
            llm_model,
        )
