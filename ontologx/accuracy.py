"""OntoLogX accuracy metrics."""

from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.scorer.scorer import Scorer
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

type _Triple = tuple[str, str, str]


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


def _node_match(node1: Node, node2: Node) -> bool:
    """Check if two nodes are equal.

    Ignores the 'id', 'uri', 'runName', and 'embedding' properties for comparison.

    Args:
        node1 (Node): First node.
        node2 (Node): Second node.

    Returns:
        bool: True if nodes are equal, False otherwise.

    """
    # Remove the 'id' and 'uri' properties from the nodes for comparison
    props1 = {
        k: (v.lower() if isinstance(v, str) else v)
        for k, v in node1.properties.items()
        if k not in {"id", "uri", "runName", "embedding"}
    }
    props2 = {
        k: (v.lower() if isinstance(v, str) else v)
        for k, v in node2.properties.items()
        if k not in {"id", "uri", "runName", "embedding"}
    }

    return node1.type == node2.type and props1 == props2


def _relationship_match(rel1: Relationship, rel2: Relationship) -> bool:
    """Check if two relationships are equal.

    Two relationships are considered equal if their types are the same and their source and target nodes match,
    using the `_node_match` function.

    Args:
        rel1 (Relationship): First relationship.
        rel2 (Relationship): Second relationship.

    Returns:
        bool: True if relationships are equal, False otherwise.

    """
    return rel1.type == rel2.type and _node_match(rel1.source, rel2.source) and _node_match(rel1.target, rel2.target)


def _llm_completeness(y_pred: list[GraphDocument], model: DeepEvalBaseLLM) -> float:
    metric = GEval(
        name="GraphCompleteness",
        model=model,
        evaluation_steps=[
            "Write a thorough description of the log event in 'input', describing what event occurred and its details.",
            "Write a thorough description of the knowledge graph in 'actual output'.",
            "Evaluate whether the knowledge graph description semantically matches the log event description.",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
        rubric=[
            Rubric(score_range=(0, 2), expected_outcome="The descriptions are not related at all."),
            Rubric(
                score_range=(2, 4),
                expected_outcome="The graph description is not related at all to the log event description.",
            ),
            Rubric(
                score_range=(4, 6),
                expected_outcome="The graph description is somewhat related to the log event description, \
                    but has omissions",
            ),
            Rubric(
                score_range=(6, 8),
                expected_outcome="The graph description is related to the log event description, \
                    and only has minor omissions.",
            ),
            Rubric(
                score_range=(8, 10),
                expected_outcome="The graph description is very related to the log event description, \
                    and has no omissions.",
            ),
        ],
    )
    test_cases = [
        LLMTestCase(
            input=graph.source.page_content if graph.source else "",
            actual_output=f'{"nodes": {graph.nodes}, "relationships": {graph.relationships}}',
        )
        for graph in y_pred
    ]

    return sum([metric.measure(test_case) for test_case in test_cases]) / len(test_cases)


def _bert_score(y_pred: list[GraphDocument]) -> float:
    """Calculate BERT score for the predicted graphs.

    Args:
        y_pred (list[GraphDocument]): List of predicted GraphDocuments.
        model (DeepEvalBaseLLM): The LLM model used for evaluation.

    Returns:
        float: The average BERT score for the predicted graphs.

    """
    references = [graph.source.page_content if graph.source else "" for graph in y_pred]
    predictions = [f'{"nodes": {graph.nodes}, "relationships": {graph.relationships}}' for graph in y_pred]
    return Scorer.bert_score(references=references, predictions=predictions)


class AccuracyEvaluator:
    """Evaluator for accuracy metrics."""

    __tp: int = 0
    """True positives, the number of triples found that are in the true set."""
    __fp: int = 0
    """False positives, the number of triples found that are not in the true set."""
    __fn: int = 0
    """False negatives, the number of triples not found that are in the true set."""
    __entities_total: int = 0
    """Total number of entities in the true set."""
    __entities_correct: int = 0
    """Number of entities correctly linked."""
    __rels_total: int = 0
    """Total number of relationships in the true set."""
    __rels_correct: int = 0

    def __init__(self, y_pred: list[GraphDocument], y_true: list[GraphDocument], llm_model: DeepEvalBaseLLM):
        """Initialize the evaluator with predictions and true values.

        Args:
            y_pred (list[GraphDocument]): List of predicted GraphDocuments.
            y_true (list[GraphDocument]): List of true GraphDocuments.
            llm_model (DeepEvalBaseLLM): The LLM model used for LLM evaluation.

        """
        self.y_pred = y_pred
        self.y_true = y_true
        self.llm_model = llm_model

        for pred, true in zip(y_pred, y_true, strict=True):
            triples_pred = _triples(pred)
            triples_true = _triples(true)

            for triple_pred in triples_pred:
                if triple_pred not in triples_true:
                    self.__fp += 1
                    continue

                self.__tp += 1
                triples_true.remove(triple_pred)

            self.__fn += len(triples_true)

            entities_pred = pred.nodes
            entities_true = true.nodes
            self.__entities_total += len(entities_true)

            rels_pred = pred.relationships
            rels_true = true.relationships
            self.__rels_total += len(rels_true)

            for entity_pred in entities_pred:
                for entity_true in entities_true:
                    if _node_match(entity_pred, entity_true):
                        self.__entities_correct += 1
                        entities_true.remove(entity_true)
                        break

            for rel_pred in rels_pred:
                for rel_true in rels_true:
                    if _relationship_match(rel_pred, rel_true):
                        self.__rels_correct += 1
                        rels_true.remove(rel_true)
                        break

    def precision(self) -> float:
        """Calculate precision score."""
        return self.__tp / (self.__tp + self.__fp) if self.__tp + self.__fp > 0 else 0

    def recall(self) -> float:
        """Calculate recall score."""
        return self.__tp / (self.__tp + self.__fn) if self.__tp + self.__fn > 0 else 0

    def f1(self) -> float:
        """Calculate F1 score."""
        precision = self.precision()
        recall = self.recall()
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    def entity_linking_accuracy(self) -> float:
        """Calculate entity linking accuracy."""
        return self.__entities_correct / self.__entities_total if self.__entities_total > 0 else 0

    def relationship_linking_accuracy(self) -> float:
        """Calculate relationship linking accuracy."""
        return self.__rels_correct / self.__rels_total if self.__rels_total > 0 else 0

    def completeness(self) -> float:
        """Calculate completeness score using LLM evaluation."""
        return _llm_completeness(self.y_pred, self.llm_model)

    def bert_score(self) -> float:
        """Calculate BERT score for the predicted graphs."""
        return _bert_score(self.y_pred)
