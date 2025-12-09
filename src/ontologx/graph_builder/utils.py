"""Utility functions for the parser module."""

import uuid

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolCall, ToolMessage
from rdflib import RDFS, Graph
from rdflib.term import URIRef


def get_messages_from_example_graph(event: str, graph: Graph) -> list[AnyMessage]:
    """Create an example message group for the given event and graph."""
    nodes = []
    relationships = []

    nodes_types = graph.subject_objects(RDFS.type)

    for node_uri, node_type in nodes_types:
        node_properties = [
            (pred, obj)
            for pred, obj in graph.predicate_objects(node_uri)
            if pred != RDFS.type and not isinstance(obj, URIRef)
        ]
        nodes.append(
            {
                "id": str(node_uri),
                "type": str(node_type),
                "properties": [{"type": str(prop), "value": str(value)} for prop, value in node_properties],
            },
        )

        node_rels = [
            (pred, obj)
            for pred, obj in graph.predicate_objects(node_uri)
            if pred != RDFS.type and isinstance(obj, URIRef)
        ]
        relationships.extend(
            {
                "source_id": str(node_uri),
                "target_id": str(target_uri),
                "type": str(pred),
            }
            for pred, target_uri in node_rels
        )

    tool_call_id = str(uuid.uuid4())
    return [
        HumanMessage(f"Event: '{event}'", name="example_user"),
        AIMessage(
            "",
            id=str(uuid.uuid4()),
            tool_calls=[
                ToolCall(
                    name="EventGraph",
                    args={
                        "nodes": nodes,
                        "relationships": relationships,
                    },
                    id=tool_call_id,
                ),
            ],
        ),
        ToolMessage("", tool_call_id=tool_call_id),
        AIMessage("Done"),
    ]
