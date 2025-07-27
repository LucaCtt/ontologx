"""Utility functions for graph documents in the Neo4j store."""

from copy import deepcopy

from ontologx.store import GraphDocument


def normalize_output_graph(graph: GraphDocument) -> GraphDocument:
    """Normalize a graph document for output by removing internal-only properties and renaming namespace separators.

    Args:
        graph (GraphDocument): The graph document to normalize.

    Returns:
        GraphDocument: The normalized graph document.

    """
    norm = deepcopy(graph)

    for node in norm.nodes:
        # Convert namespace separator for node
        node.type = node.type.replace("n4sch", "rdfs").replace("__", ":")

        # Convert namespace separator for properties, rename "n4sch" to "rdfs",
        for key in list(node.properties.keys()):
            # Remove internal-only properties
            if key in ["uri", "embedding"]:
                node.properties.pop(key)
                continue

            new_key = key.replace("n4sch", "rdfs").replace("__", ":")
            node.properties[new_key] = node.properties.pop(key)

    for relationship in norm.relationships:
        # Convert namespace separator for relationship type
        relationship.type = relationship.type.replace("n4sch", "rdfs").replace("__", ":")

        # Convert namespace separator for properties, rename "n4sch" to "rdfs"
        for key in list(relationship.properties.keys()):
            new_key = key.replace("n4sch", "rdfs").replace("__", ":")
            relationship.properties[new_key] = relationship.properties.pop(key)

    return norm


def normalize_input_graph(graph: GraphDocument) -> GraphDocument:
    """Normalize a graph document for input by converting namespace separators and renaming properties.

    Args:
        graph (GraphDocument): The graph document to normalize.

    Returns:
        GraphDocument: The normalized graph document.

    """
    # Create a new normalized graph document, so the original is not modified
    norm = deepcopy(graph)

    for node in norm.nodes:
        # Copy node without properties for normalization
        node.type = node.type.replace("rdfs", "n4sch").replace(":", "__")

        # Insert uri property
        node.properties["uri"] = node.id

        # Convert namespace separator for properties, rename "rdfs" to "n4sch"
        for key in list(node.properties.keys()):
            new_key = key.replace("rdfs", "n4sch").replace(":", "__")
            node.properties[new_key] = node.properties.pop(key)

    for relationship in norm.relationships:
        # Convert namespace separator for relationship type
        relationship.type = relationship.type.replace("rdfs", "n4sch").replace(":", "__")

        # Convert namespace separator for properties, rename "rdfs" to "n4sch"
        for key in list(relationship.properties.keys()):
            new_key = key.replace("rdfs", "n4sch").replace(":", "__")
            relationship.properties[new_key] = relationship.properties.pop(key)

    return norm
