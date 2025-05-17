"""OntoLogX accuracy metrics."""

from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

type Triple = tuple[str, str, str]


def __triples(graph: GraphDocument) -> list[Triple]:
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


def __node_match(node1: Node, node2: Node) -> bool:
    """Check if two nodes are equal.

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


def __relationship_match(rel1: Relationship, rel2: Relationship) -> bool:
    """Check if two relationships are equal.

    Args:
        rel1 (Relationship): First relationship.
        rel2 (Relationship): Second relationship.

    Returns:
        bool: True if relationships are equal, False otherwise.

    """
    return rel1.type == rel2.type and __node_match(rel1.source, rel2.source) and __node_match(rel1.target, rel2.target)


def relevancy(y_pred: list[GraphDocument]) -> float:
    correctness = GEval(
        name="GraphRelevance",
        evaluation_steps=[
            "Evaluate whether the triples in 'actual output' are relevant and faithul to the log event in 'input'.",
            "Check if the entities in the triples "
            "Check if the relationships between nodes are meaningful and accurately reflect the log event.",
            "Do not evaluate the 'id' values, which are useful only to define relationships.",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
        rubric=[
            Rubric(score_range=(0, 2), expected_outcome="Graph does not reflect the log at all."),
            Rubric(
                score_range=(2, 4),
                expected_outcome="Graph mostly wrong or incomplete.",
            ),
            Rubric(
                score_range=(4, 6),
                expected_outcome="Graph is mostly correct but has some errors.",
            ),
            Rubric(
                score_range=(6, 8),
                expected_outcome="Graph has only small omissions or mistakes.",
            ),
            Rubric(
                score_range=(8, 10),
                expected_outcome="Graph is correct and faithful.",
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

    return sum([correctness.measure(test_case) for test_case in test_cases]) / len(test_cases)


def metrics(y_pred: list[GraphDocument], y_true: list[GraphDocument]) -> tuple[float, float, float, float, float]:
    """Calculate evalutation metrics for the predictions.

    Args:
        y_pred (list[GraphDocument]): List of predicted GraphDocuments.
        y_true (list[GraphDocument]): List of true GraphDocuments.

    Returns:
        float: Precision score.
        float: Recall score.
        float: F1 score.
        float: Entity linking accuracy score.
        float: Relationship linking accuracy score.

    """
    graphs_pred = [__triples(graph) for graph in y_pred]
    graphs_true = [__triples(graph) for graph in y_true]

    tp = 0  # triples found that are in the true set
    fp = 0  # triples found that are not in the true set
    fn = 0  # triples not found that are in the true set

    for i in range(len(graphs_pred)):
        triples_graph_pred = graphs_pred[i]
        triples_graph_true = graphs_true[i]

        for triple_pred in triples_graph_pred:
            if triple_pred not in triples_graph_true:
                fp += 1
                continue

            tp += 1
            triples_graph_true.remove(triple_pred)

        fn += len(triples_graph_true)

    graphs_entities_pred = [graph.nodes for graph in y_pred]
    graphs_entities_true = [graph.nodes for graph in y_true]
    entities_total = sum(len(nodes) for nodes in graphs_entities_true)
    entities_correct = 0
    for entities_pred, entities_true in zip(graphs_entities_pred, graphs_entities_true, strict=True):
        for entity_pred in entities_pred:
            for entity_true in entities_true:
                if __node_match(entity_pred, entity_true):
                    entities_correct += 1
                    entities_true.remove(entity_true)
                    break

    graphs_rels_pred = [graph.relationships for graph in y_pred]
    graphs_rels_true = [graph.relationships for graph in y_true]
    rels_total = sum(len(relationships) for relationships in graphs_rels_true)
    rels_correct = 0
    for rels_pred, rels_true in zip(
        graphs_rels_pred,
        graphs_rels_true,
        strict=True,
    ):
        for rel_pred in rels_pred:
            for rel_true in rels_true:
                if __relationship_match(rel_pred, rel_true):
                    rels_correct += 1
                    rels_true.remove(rel_true)
                    break

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    ela = entities_correct / entities_total if entities_total > 0 else 0
    rla = rels_correct / rels_total if rels_total > 0 else 0

    return precision, recall, f1, ela, rla
