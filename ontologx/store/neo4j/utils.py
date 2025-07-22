from ontologx.store import GraphDocument


def normalize_output_graph(graph: GraphDocument) -> GraphDocument:
    """Normalize a graph document for output by removing internal-only properties and renaming namespace separators.

    Args:
        graph (GraphDocument): The graph document to normalize.

    Returns:
        GraphDocument: The normalized graph document.

    """
    for node in graph.nodes:
        # Remove embedding property
        node.properties.pop("n4sch__embedding", None)

        # Convert namespace separator for node
        node.type = node.type.replace("__", ":")

        # Convert namespace separator for properties, rename "n4sch" to "schema"
        for key in list(node.properties.keys()):
            new_key = key.replace("n4sch", "schema").replace("__", ":")
            node.properties[new_key] = node.properties.pop(key)

    for relationship in graph.relationships:
        # Convert namespace separator for relationship type
        relationship.type = relationship.type.replace("__", ":")

        # Convert namespace separator for properties, rename "n4sch" to "schema"
        for key in list(relationship.properties.keys()):
            new_key = key.replace("n4sch", "schema").replace("__", ":")
            relationship.properties[new_key] = relationship.properties.pop(key)

    return graph


def normalize_input_graph(graph: GraphDocument) -> GraphDocument:
    """Normalize a graph document for input by converting namespace separators and renaming properties.

    Args:
        graph (GraphDocument): The graph document to normalize.

    Returns:
        GraphDocument: The normalized graph document.

    """
    for node in graph.nodes:
        # Convert namespace separator for node
        node.type = node.type.replace(":", "__")

        # Convert namespace separator for properties, rename "schema" to "n4sch"
        for key in list(node.properties.keys()):
            new_key = key.replace(":", "__").replace("schema", "n4sch")
            node.properties[new_key] = node.properties.pop(key)

    for relationship in graph.relationships:
        # Convert namespace separator for relationship type
        relationship.type = relationship.type.replace(":", "__")

        # Convert namespace separator for properties, rename "schema" to "n4sch"
        for key in list(relationship.properties.keys()):
            new_key = key.replace(":", "__").replace("schema", "n4sch")
            relationship.properties[new_key] = relationship.properties.pop(key)

    return graph
