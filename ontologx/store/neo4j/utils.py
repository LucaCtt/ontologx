from ontologx.store import GraphDocument, Node, Relationship


def normalize_output_graph(graph: GraphDocument) -> GraphDocument:
    """Normalize a graph document for output by removing internal-only properties and renaming namespace separators.

    Args:
        graph (GraphDocument): The graph document to normalize.

    Returns:
        GraphDocument: The normalized graph document.

    """
    for node in graph.nodes:
        # Convert namespace separator for node
        node.type = node.type.replace("__", ":")

        # Remove embedding property
        node.properties.pop("n4sch__embedding", None)

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
    # Create a new normalized graph document, so the original is not modified
    norm = GraphDocument(nodes=[], relationships=[], source=graph.source)

    nodes_dict = {}
    for node in graph.nodes:
        # Copy node without properties for normalization
        node_norm = Node(id=node.id, type=node.type.replace(":", "__"), properties={})

        # Convert namespace separator for properties, rename "schema" to "n4sch"
        for key in list(node.properties.keys()):
            new_key = key.replace("schema", "n4sch").replace(":", "__")
            node_norm.properties[new_key] = node.properties[key]

        nodes_dict[node.id] = node_norm

    norm.nodes = list(nodes_dict.values())

    for relationship in graph.relationships:
        # Convert namespace separator for relationship type
        rel_norm = Relationship(
            type=relationship.type.replace(":", "__"),
            source=nodes_dict[relationship.source.id],
            target=nodes_dict[relationship.target.id],
            properties={},
        )

        # Convert namespace separator for properties, rename "schema" to "n4sch"
        for key in list(rel_norm.properties.keys()):
            new_key = key.replace("schema", "n4sch").replace(":", "__")
            rel_norm.properties[new_key] = relationship.properties[key]

    return norm
