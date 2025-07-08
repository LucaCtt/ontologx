"""OntoLogX accuracy metrics."""

from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams

from ontologx.store import GraphDocument, Node, Relationship

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
            if prop in ("id", "uri", "runName", "embedding"):
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
        if k not in {"id", "uri", "runName", "embedding"}
    }
    props2 = {
        k: (v.lower() if isinstance(v, str) else v)
        for k, v in entity2.properties.items()
        if k not in {"id", "uri", "runName", "embedding"}
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


def _geval(y_pred: list[GraphDocument], model: DeepEvalBaseLLM) -> float:
    metric = GEval(
        name="Alignment",
        model=model,
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
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    )
    test_cases = [
        LLMTestCase(
            input=graph.source.page_content,
            actual_output=f"{_triples_without_event_message(graph)}",
        )
        for graph in y_pred
    ]

    measures = []
    for test_case in test_cases:
        # Re-run the measure for 5 attempts to mitigate invalid LLM output issues
        res = 0
        for _ in range(5):
            try:
                res = metric.measure(test_case, _show_indicator=False)
                break
            except ValueError as _:
                continue

        measures.append(res)

    return sum(measures) / len(test_cases)


class OntologyGraphMetrics:
    """Metrics for evaluating ontology graphs.

    This class computes precision, recall, F1 score, entity linking accuracy,
    relationship linking accuracy, using a given predicted and true GraphDocument.
    """

    precision: float = 0.0
    recall: float = 0.0
    entity_linking_accuracy: float = 0.0
    relationship_linking_accuracy: float = 0.0

    @property
    def f1(self) -> float:
        """Calculate F1 score based on precision and recall."""
        return (
            2 * (self.precision * self.recall) / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0
        )

    def __init__(self, pred: GraphDocument, true: GraphDocument):
        """Initialize the evaluator with predictions and true values.

        Args:
            pred (GraphDocument): List of predicted GraphDocuments.
            true (GraphDocument): List of true GraphDocuments.

        """
        tp = 0  # True positives, the number of triples found that are in the true set
        fp = 0  # False positives, the number of triples found that are not in the true set
        fn = 0  # False negatives, the number of triples not found that are in the true set

        triples_pred = _triples(pred)
        triples_true = _triples(true)

        for triple_pred in triples_pred:
            if triple_pred not in triples_true:
                fp += 1
                continue

            tp += 1
            triples_true.remove(triple_pred)

        fn += len(triples_true)

        self.precision = tp / (tp + fp) if tp + fp > 0 else 0
        self.recall = tp / (tp + fn) if tp + fn > 0 else 0

        entities_correct = []
        entities_pred = pred.nodes.copy()
        entities_true = true.nodes.copy()
        for entity_pred in entities_pred:
            for entity_true in entities_true:
                if _entity_match(entity_pred, entity_true):
                    entities_correct.append(entity_pred)
                    entities_true.remove(entity_true)
                    break
        self.entity_linking_accuracy = len(entities_correct) / len(true.nodes) if entities_true else 0

        # Only consider relatioships among correct entities
        candidate_rels = [
            r for r in pred.relationships if r.source in entities_correct and r.target in entities_correct
        ]
        rels_correct = 0
        rels_true = true.relationships.copy()
        for rel_pred in candidate_rels:
            for rel_true in rels_true:
                if _relationship_match(rel_pred, rel_true):
                    rels_correct += 1
                    rels_true.remove(rel_true)
                    break
        self.relationship_linking_accuracy = rels_correct / len(candidate_rels) if candidate_rels else 0


class AccuracyEvaluator:
    """Evaluator for accuracy metrics."""

    @property
    def precision(self) -> float:
        """Calculate precision across all metrics."""
        return sum(metric.precision for metric in self.metrics) / len(self.metrics)

    @property
    def recall(self) -> float:
        """Calculate recall across all metrics."""
        return sum(metric.recall for metric in self.metrics) / len(self.metrics)

    @property
    def f1(self) -> float:
        """Calculate F1 score across all metrics."""
        return sum(metric.f1 for metric in self.metrics) / len(self.metrics)

    @property
    def entity_linking_accuracy(self) -> float:
        """Calculate entity linking accuracy across all metrics."""
        return sum(metric.entity_linking_accuracy for metric in self.metrics) / len(self.metrics)

    @property
    def relationship_linking_accuracy(self) -> float:
        """Calculate relationship linking accuracy across all metrics."""
        return sum(metric.relationship_linking_accuracy for metric in self.metrics) / len(self.metrics)

    @property
    def geval(self) -> float:
        """Calculate G-Eval score across all metrics using the provided LLM model.

        Returns:
            float: The average G-Eval score across all metrics.

        """
        return _geval(self.__y_pred, self.__llm_model)

    def __init__(self, y_pred: list[GraphDocument], y_true: list[GraphDocument], llm_model: DeepEvalBaseLLM):
        """Initialize the evaluator with predictions and true values.

        Args:
            y_pred (list[GraphDocument]): List of predicted GraphDocuments.
            y_true (list[GraphDocument]): List of true GraphDocuments.
            llm_model (DeepEvalBaseLLM): The LLM model used for LLM evaluation.

        """
        self.metrics = [OntologyGraphMetrics(pred, true) for pred, true in zip(y_pred, y_true, strict=True)]
        self.__llm_model = llm_model
        self.__y_pred = y_pred
