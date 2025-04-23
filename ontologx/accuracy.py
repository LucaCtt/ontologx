"""Parser accuracy metrics."""

import networkx as nx
from langchain_neo4j.graphs.graph_document import GraphDocument


def __graph_document_to_networkx(graph: GraphDocument) -> nx.Graph:
    g = nx.Graph()
    for node in graph.nodes:
        g.add_node(node.type)
        for prop_name, prop_value in node.properties.items():
            g.add_node(prop_name)
            g.add_edge(node.type, prop_name, label=prop_value)

    for relationship in graph.relationships:
        g.add_edge(relationship.source.type, relationship.target.type, label=relationship.type)
    return g


def graph_edit_distance(graph1: GraphDocument, graph2: GraphDocument) -> float:
    gn1 = __graph_document_to_networkx(graph1)
    gn2 = __graph_document_to_networkx(graph2)

    # Pretty sketchy, not sure if this is the right way to do this
    def node_match(n1, n2) -> bool:
        return n1 == n2

    def edge_match(e1, e2) -> bool:
        return e1["label"] == e2["label"] and e1[0] == e2[0] and e1[1] == e2[1]

    return nx.graph_edit_distance(gn1, gn2, node_match=node_match, edge_match=edge_match)
