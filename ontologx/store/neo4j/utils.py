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
        node.type = node.type.replace("__", ":")

        # Remove internal-onlt property
        node.properties.pop("n4sch__embedding", None)
        node.properties.pop("n4sch__runName", None)
        node.properties.pop("mls__implements", None)

        # Convert namespace separator for properties, rename "n4sch" to "schema",
        for key in list(node.properties.keys()):
            new_key = key.replace("n4sch", "schema").replace("__", ":")
            node.properties[new_key] = node.properties.pop(key)

    for relationship in norm.relationships:
        # Convert namespace separator for relationship type
        relationship.type = relationship.type.replace("__", ":")

        # Convert namespace separator for properties, rename "n4sch" to "schema"
        for key in list(relationship.properties.keys()):
            new_key = key.replace("n4sch", "schema").replace("__", ":")
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
        node.type = node.type.replace(":", "__")

        # Convert namespace separator for properties, rename "schema" to "n4sch"
        for key in list(node.properties.keys()):
            new_key = key.replace("schema", "n4sch").replace(":", "__")
            node.properties[new_key] = node.properties.pop(key)

    for relationship in norm.relationships:
        # Convert namespace separator for relationship type
        relationship.type = relationship.type.replace(":", "__")

        # Convert namespace separator for properties, rename "schema" to "n4sch"
        for key in list(relationship.properties.keys()):
            new_key = key.replace("schema", "n4sch").replace(":", "__")
            relationship.properties[new_key] = relationship.properties.pop(key)

    return norm
