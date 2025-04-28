"""OntoLogX accuracy metrics."""

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
            if prop in ("id", "uri", "runName"):
                continue
            triples.append((node.id, prop, value))

    triples.extend([(rel.source.id, rel.type, rel.target.id) for rel in graph.relationships])

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
    triples_pred = [__triples(graph) for graph in y_pred]
    triples_true = [__triples(graph) for graph in y_true]

    tp = 0  # triples found that are in the true set
    fp = 0  # triples found that are not in the true set
    fn = 0  # triples not found that are in the true set

    for triple_pred in triples_pred:
        if triple_pred not in triples_true:
            fp += 1
            continue

        tp += 1
        triples_true.remove(triple_pred)

    fn += len(triples_true)

    entities_pred = [node.type for graph in y_pred for node in graph.nodes]
    entities_true = [node.type for graph in y_true for node in graph.nodes]
    entities_total = len(entities_true)
    entities_correct = 0
    for entity_pred in entities_pred:
        if entity_pred in entities_true:
            entities_correct += 1
            entities_true.remove(entity_pred)

    relationships_pred = [rel.type for graph in y_pred for rel in graph.relationships]
    relationships_true = [rel.type for graph in y_true for rel in graph.relationships]
    relationships_total = len(relationships_true)
    relationships_correct = 0
    for relationship_pred in relationships_pred:
        if relationship_pred in relationships_true:
            relationships_correct += 1
            relationships_true.remove(relationship_pred)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    ela = entities_correct / entities_total if entities_total > 0 else 0
    rla = relationships_correct / relationships_total if relationships_total > 0 else 0

    return precision, recall, f1, ela, rla
