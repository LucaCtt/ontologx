"""OntoLogX accuracy metrics."""

from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship


def __node_match(node1: Node, node2: Node) -> bool:
    """Check if two nodes are equal.

    Args:
        node1 (Node): First node.
        node2 (Node): Second node.

    Returns:
        bool: True if nodes are equal, False otherwise.

    """
    # Remove the 'id' and 'uri' properties from the nodes for comparison
    props1 = {k: v for k, v in node1.properties.items() if k not in {"id", "uri"}}
    props2 = {k: v for k, v in node2.properties.items() if k not in {"id", "uri"}}

    return node1.type == node2.type and props1 == props2


def __relationship_match(rel1: Relationship, rel2: Relationship) -> bool:
    """Check if two relationships are equal.

    Args:
        rel1 (Relationship): First relationship.
        rel2 (Relationship): Second relationship.

    Returns:
        bool: True if relationships are equal, False otherwise.

    """
    # Remove the 'id' property from the relationships for comparison
    props1 = {k: v for k, v in rel1.properties.items() if k != "id"}
    props2 = {k: v for k, v in rel2.properties.items() if k != "id"}

    return (
        rel1.type == rel2.type
        and props1 == props2
        and __node_match(rel1.source, rel2.source)
        and __node_match(rel1.target, rel2.target)
    )


def metrics(y_pred: list[GraphDocument], y_true: list[GraphDocument]) -> tuple[float, float, float]:
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
    nodes_correct = 0
    nodes_total = 0

    relationships_correct = 0
    relationships_total = 0

    all_correct = 0

    for pred, true in zip(y_pred, y_true, strict=True):
        nodes_total += len(pred.nodes)
        nodes_correct += sum(
            1 for pred_node in pred.nodes if any(__node_match(pred_node, true_node) for true_node in true.nodes)
        )

        relationships_total += len(pred.relationships)
        relationships_correct += sum(
            1 for pred_edge in pred.edges if any(__relationship_match(pred_edge, true_edge) for true_edge in true.edges)
        )

        if (
            len(pred.nodes) == len(true.nodes)
            and len(pred.edges) == len(true.edges)
            and nodes_correct == len(pred.nodes)
            and relationships_correct == len(pred.edges)
        ):
            all_correct += 1

    precision = all_correct / len(y_pred)
    recall = all_correct / len(y_true)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    ela = nodes_correct / nodes_total if nodes_total > 0 else 0
    rla = relationships_correct / relationships_total if relationships_total > 0 else 0

    return precision, recall, f1, ela, rla
