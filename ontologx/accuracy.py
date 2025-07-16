"""OntoLogX accuracy metrics."""

import functools

from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams

from ontologx.store import GraphDocument, Node, Relationship
from ontologx.store.store import Store

type _Triple = tuple[str, str, str]  # Triple in the form (subject, predicate, object)


def _triples(graph: GraphDocument) -> list[_Triple]:
    """Get the triples from a graph.

    Args:
        graph (GraphDocument): The graph to extract triples from.

    Returns:
        list[tuple[str,str,str]]: A list of triples in the form (subject, predicate, object).

    """
    triples = []
    for node in graph.nodes:
        for prop, value in node.properties.items():
            if prop in ("id", "uri") or prop.startswith("n4sch"):
                continue
            triples.append((node.type, prop, value))

    triples.extend([(rel.source.type, rel.type, rel.target.type) for rel in graph.relationships])

    return triples


def _triples_without_event_message(graph: GraphDocument) -> list[_Triple]:
    """Get the triples from a graph without filtering.

    Args:
        graph (GraphDocument): The graph to extract triples from.

    Returns:
        list[tuple[str,str,str]]: A list of triples in the form (subject, predicate, object),
                                  excluding 'eventMessage' predicates.

    """
    return [t for t in _triples(graph) if t[1] != "eventMessage"]


def _entity_match(entity1: Node, entity2: Node) -> bool:
    """Check if two entities are equal.

    Two entities are considered equal if their types and values match.

    Args:
        entity1 (Node): First entity.
        entity2 (Node): Second entity.

    Returns:
        bool: True if entities are equal, False otherwise.

    """
    props1 = {
        k: (v.lower() if isinstance(v, str) else v)
        for k, v in entity1.properties.items()
        if (k not in {"id", "uri"} and not k.startswith("n4sch"))  # Exclude id, uri, and n4sch properties
    }
    props2 = {
        k: (v.lower() if isinstance(v, str) else v)
        for k, v in entity2.properties.items()
        if (k not in {"id", "uri"} and not k.startswith("n4sch"))  # Exclude id, uri, and n4sch properties
    }

    return entity1.type == entity2.type and props1 == props2


def _relationship_match(rel1: Relationship, rel2: Relationship) -> bool:
    """Check if two entity relationships are equal.

    Two entity relationships are considered equal if their entities and relationship type match,
    using the `_entity_match` function.

    Args:
        rel1 (Relationship): First entity relationship.
        rel2 (Relationship): Second entity relationship.

    Returns:
        bool: True if entity relationships are equal, False otherwise.

    """
    return (
        rel1.type == rel2.type and _entity_match(rel1.source, rel2.source) and _entity_match(rel1.target, rel2.target)
    )


class OntologyGraphMetrics:
    """Metrics for evaluating ontology graphs.

    This class computes precision, recall, F1 score, entity linking accuracy,
    relationship linking accuracy, using a given predicted and true GraphDocument.
    """

    __tp = 0  # True positives, the number of triples found that are in the true set
    __fp = 0  # False positives, the number of triples found that are not in the true set
    __fn = 0  # False negatives, the number of triples not found that are in the true set

    @functools.cached_property
    def precision(self) -> float:
        """Calculate precision based on true positives and false positives."""
        return self.__tp / (self.__tp + self.__fp) if (self.__tp + self.__fp) > 0 else 0

    @functools.cached_property
    def recall(self) -> float:
        """Calculate recall based on true positives and false negatives."""
        return self.__tp / (self.__tp + self.__fn) if (self.__tp + self.__fn) > 0 else 0

    @functools.cached_property
    def f1(self) -> float:
        """Calculate F1 score based on precision and recall."""
        return (
            2 * (self.precision * self.recall) / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0
        )

    @functools.cached_property
    def __entities_correct(self) -> list[Node]:
        """Get the list of correctly matched entities."""
        entities_correct = []
        entities_pred = self.__pred.nodes.copy()
        entities_true = self.__true.nodes.copy()
        for entity_pred in entities_pred:
            for entity_true in entities_true:
                if _entity_match(entity_pred, entity_true):
                    entities_correct.append(entity_pred)
                    entities_true.remove(entity_true)
                    break

        return entities_correct

    @functools.cached_property
    def entity_linking_accuracy(self) -> float:
        """Calculate entity linking accuracy based on correct entities."""
        return len(self.__entities_correct) / len(self.__true.nodes)

    @functools.cached_property
    def relationship_linking_accuracy(self) -> float:
        """Calculate relationship linking accuracy based on correct relationships."""
        # Only consider relatioships among correct entities
        candidate_rels = [
            r
            for r in self.__pred.relationships
            if r.source in self.__entities_correct and r.target in self.__entities_correct
        ]

        rels_correct = 0
        rels_true = self.__true.relationships.copy()
        for rel_pred in candidate_rels:
            for rel_true in rels_true:
                if _relationship_match(rel_pred, rel_true):
                    rels_correct += 1
                    rels_true.remove(rel_true)
                    break
        return rels_correct / len(candidate_rels) if candidate_rels else 0

    @functools.cached_property
    def shacl_violations_percentage(self) -> float:
        """Calculate the percentage of SHACL violations in the predicted graph."""
        return self.__store.validate_event_graph(self.__pred) / self.__store.total_constraints()

    def __init__(self, pred: GraphDocument, true: GraphDocument, store: Store):
        """Initialize the evaluator with predictions and true values.

        Args:
            pred (GraphDocument): List of predicted GraphDocuments.
            true (GraphDocument): List of true GraphDocuments.
            store (Store): The store used to manage GraphDocuments.

        """
        self.__pred = pred
        self.__true = true
        self.__store = store

        triples_pred = _triples(self.__pred)
        triples_true = _triples(self.__true)

        for triple_pred in triples_pred:
            if triple_pred not in triples_true:
                self.__fp += 1
                continue

            self.__tp += 1
            triples_true.remove(triple_pred)

        self.__fn += len(triples_true)


class GEvalMetrics:
    @functools.cached_property
    def __measures(self) -> list[float]:
        result = []
        for test_case in self.__test_cases:
            # If the actual output is None, skip the LLM evaluation completely
            if test_case.actual_output == "[]":
                result.append(0)
                continue

            # Re-run the measure for 5 attempts to mitigate invalid LLM output issues
            res = 0
            for _ in range(5):
                try:
                    res = self.__metric.measure(test_case, _show_indicator=False)
                    break
                except ValueError as _:
                    continue

            result.append(res)

        return result

    @functools.cached_property
    def mean(self) -> float:
        """Calculate the mean of the G-Eval measures."""
        return sum(self.__measures) / len(self.__measures) if self.__measures else 0

    @functools.cached_property
    def mean_valid_only(self) -> float:
        """Calculate the mean of the G-Eval measures, excluding invalid outputs."""
        return (
            sum(
                measure
                for measure, graph in zip(self.__measures, self.__y_pred, strict=True)
                if self.__store.validate_event_graph(graph) == 0
            )
            / len(self.__measures)
            if self.__measures
            else 0
        )

    def __init__(self, y_pred: list[GraphDocument], llm_model: DeepEvalBaseLLM, store: Store):
        """Initialize the G-Eval metrics with predictions and the LLM model.

        Args:
            y_pred (list[GraphDocument]): List of predicted GraphDocuments.
            llm_model (DeepEvalBaseLLM): The LLM model used for G-Eval.
            store (Store): The store used to manage GraphDocuments.

        """
        self.__y_pred = y_pred
        self.__store = store

        self.__metric = GEval(
            name="Graph Alignment",
            model=llm_model,
            evaluation_steps=[
                (
                    "Write a detailed description of the input log event in natural language. "
                    "Include what occurred, the involved entities, their roles, any parameters, "
                    "timestamps, or contextual details conveyed in the log."
                ),
                (
                    "Translate the output knowledge graph, expressed as triples (subject-predicate-object), "
                    "into a coherent natural language description that reflects the event(s) they represent."
                ),
                (
                    "Assess whether the graph description semantically captures "
                    "the same core information as the log event. "
                    "Check for:\n"
                    "  - Coverage: Are all key elements from the log event present?\n"
                    "  - Correctness: Are entities, actions, and relationships represented accurately?\n"
                    "  - Relevance: Are any additional triples relevant to the log event context?\n"
                    "It is acceptable if the graph contains more information than the log event, "
                    "as long as it enriches the representation without introducing unrelated or incorrect content."
                ),
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        )

        self.__test_cases = [
            LLMTestCase(
                input=graph.source.page_content,
                actual_output=f"{_triples_without_event_message(graph)}",
            )
            for graph in y_pred
        ]


class AccuracyEvaluator:
    """Evaluator for accuracy metrics."""

    @functools.cached_property
    def shacl_violations_percentage(self) -> float:
        """Calculate the percentage of SHACL violations across all metrics."""
        return sum(metric.shacl_violations_percentage for metric in self.__metrics) / len(self.__metrics)

    @functools.cached_property
    def precision(self) -> float:
        """Calculate precision across all metrics."""
        return sum(metric.precision for metric in self.__metrics) / len(self.__metrics)

    @functools.cached_property
    def recall(self) -> float:
        """Calculate recall across all metrics."""
        return sum(metric.recall for metric in self.__metrics) / len(self.__metrics)

    @functools.cached_property
    def f1(self) -> float:
        """Calculate F1 score across all metrics."""
        return sum(metric.f1 for metric in self.__metrics) / len(self.__metrics)

    @functools.cached_property
    def entity_linking_accuracy(self) -> float:
        """Calculate entity linking accuracy across all metrics."""
        return sum(metric.entity_linking_accuracy for metric in self.__metrics) / len(self.__metrics)

    @functools.cached_property
    def relationship_linking_accuracy(self) -> float:
        """Calculate relationship linking accuracy across all metrics."""
        return sum(metric.relationship_linking_accuracy for metric in self.__metrics) / len(self.__metrics)

    @functools.cached_property
    def geval_mean(self) -> float:
        """Calculate G-Eval score across all metrics using the provided LLM model."""
        return self.__geval_metrics.mean

    @functools.cached_property
    def geval_mean_valid_only(self) -> float:
        """Calculate G-Eval score across all metrics, excluding invalid outputs."""
        return self.__geval_metrics.mean_valid_only

    def __init__(
        self,
        y_pred: list[GraphDocument],
        y_true: list[GraphDocument],
        llm_model: DeepEvalBaseLLM,
        store: Store,
    ):
        """Initialize the evaluator with predictions and true values.

        Args:
            y_pred (list[GraphDocument]): List of predicted GraphDocuments.
            y_true (list[GraphDocument]): List of true GraphDocuments.
            llm_model (DeepEvalBaseLLM): The LLM model used for LLM evaluation.
            store (Store): The store used to manage GraphDocuments.

        """
        self.__metrics = [OntologyGraphMetrics(pred, true, store) for pred, true in zip(y_pred, y_true, strict=True)]
        self.__geval_metrics = GEvalMetrics(y_pred, llm_model, store)
