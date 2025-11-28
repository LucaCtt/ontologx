"""Utility functions for the parser module."""

import uuid

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolCall, ToolMessage

from ontologx.store import GraphDocument


def get_messages_from_example_graph(graph: GraphDocument) -> list[AnyMessage]:
    """Create an example message group for the given event and graph."""
    event = graph.source.page_content

    nodes = [
        {
            "id": node.id,
            "type": node.type,
            "properties": [{"type": key, "value": value} for key, value in node.properties.items()],
        }
        for node in graph.nodes
    ]

    relationships = [
        {
            "source_id": rel.source.id,
            "target_id": rel.target.id,
            "type": rel.type,
        }
        for rel in graph.relationships
    ]

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
