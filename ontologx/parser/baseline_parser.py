import json

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

from ontologx.parser.parser import Parser
from ontologx.store import Store


class BaselineParser(Parser):
    """Baseline class asking a LLM to create a KG, without any improvement."""

    def __init__(self, llm: BaseChatModel, store: Store, prompt_build_graph: str) -> None:
        super().__init__(llm, store, prompt_build_graph)

        gen_graph_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_build_graph),
                ("human", "Event: '{event}'\nContext: '{context}'"),
            ],
        )

        self.chain = gen_graph_prompt | llm

    def parse(self, event: str, context: dict) -> GraphDocument | None:
        """Parse the given event and construct a knowledge graph, without using tools.

        Args:
            event: The log event to parse.
            context: The context of the event.

        Returns:
            A report containing the stats of the parsing process.

        """
        out = self.chain.invoke({"event": event, "context": context})

        # If not using tools, the output is hopefully in json format
        raw_schema = json.loads(out)

        if "nodes" not in raw_schema or not isinstance(raw_schema["nodes"], list):
            return None

        output_graph = GraphDocument(
            nodes=[],
            relationships=[],
            raw_output=raw_schema,
        )
        nodes_dict = {}

        for node in raw_schema["nodes"]:
            if not isinstance(node, dict):
                continue

            node_type = node.get("type", None)
            node_properties = node.get("properties", {})
            node_uri = node_properties.get("uri", None)
            if not node_uri or not node_type:
                continue

            nodes_dict.add(Node(node_type, node_type, node_properties))

        output_graph.nodes = list(nodes_dict.values())

        if "relationships" not in raw_schema or not isinstance(raw_schema["relationships"], list):
            return output_graph

        for relationship in raw_schema["relationships"]:
            if not isinstance(relationship, dict):
                continue

            start_node_id = relationship.get("startNode", None)
            end_node_id = relationship.get("endNode", None)
            rel_type = relationship.get("type", None)

            if not start_node_id or not end_node_id or not rel_type:
                continue

            start_node = nodes_dict.get(start_node_id)
            end_node = nodes_dict.get(end_node_id)
            if not start_node or not end_node:
                continue

            output_graph.add_relationship(
                Relationship(
                    start_node,
                    end_node,
                    rel_type,
                ),
            )

        return output_graph
