"""Baseline parser for constructing a knowledge graph using a language model with plain prompting."""

import json
import re

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from ontologx.parser.models import build_baseline_prompt
from ontologx.parser.parser import Parser
from ontologx.store import GraphDocument, Node, Relationship, Store


def _parse_json(llm_output: BaseMessage) -> dict:
    """Parse the LLM output as JSON."""
    try:
        return json.loads(
            llm_output.content.replace("\n", "")
            if isinstance(llm_output.content, str)
            else json.dumps(llm_output.content),
        )
    except json.JSONDecodeError:
        # If the output is not in pure JSON format, check if any output is present
        # between <output> tags (for reasoning models).
        if isinstance(llm_output.content, str):
            # Candidate output JSON
            candidates = re.findall(r"\{.*?\}", llm_output.content, re.DOTALL)

            # Get the longest candidates first, as they are more likely to be complete JSON objects
            candidates.sort(key=len, reverse=True)

            for candidate in candidates:
                try:
                    # Attempt to parse the candidate as JSON
                    out = json.loads(candidate.replace("\n", ""))

                    # Check if the output contains both nodes and relationships
                    if out.get("nodes") and out.get("relationships"):
                        return out
                except json.JSONDecodeError:
                    # If the candidate is not valid JSON, continue to the next one
                    continue

        # If nothing else works, return an empty dictionary
        return {}


class BaselineParser(Parser):
    """Baseline class asking a LLM to create a KG, without any improvement."""

    def __init__(self, llm: BaseChatModel, store: Store, prompt_build_graph: str) -> None:
        super().__init__(llm, store, prompt_build_graph)

        prompt = build_baseline_prompt(self.store.ontology(), prompt_build_graph)

        gen_graph_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
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
        raw_schema = _parse_json(out)

        output_graph = GraphDocument(
            nodes=[],
            relationships=[],
            source=Document(page_content=event, metadata={"context": context}),
        )

        if "nodes" not in raw_schema or not isinstance(raw_schema["nodes"], list):
            return output_graph

        nodes_dict = {}
        for node in raw_schema["nodes"]:
            if not isinstance(node, dict):
                continue

            node_id = node.get("id", None)
            node_type = node.get("type", None)
            node_properties = node.get("properties", [])
            node_properties = (
                {prop["type"]: prop.get("value") for prop in node_properties if prop.get("type") is not None}
                if node_properties
                else {}
            )
            if not node_id or not node_type:
                continue

            nodes_dict[node_id] = Node(id=node_type, type=node_type, properties=node_properties)

        output_graph.nodes.extend(nodes_dict.values())

        if "relationships" not in raw_schema or not isinstance(raw_schema["relationships"], list):
            return output_graph

        for relationship in raw_schema["relationships"]:
            if not isinstance(relationship, dict):
                continue

            start_node_id = relationship.get("source_id", None)
            end_node_id = relationship.get("target_id", None)
            rel_type = relationship.get("type", None)

            if not start_node_id or not end_node_id or not rel_type:
                continue

            start_node = nodes_dict.get(start_node_id)
            end_node = nodes_dict.get(end_node_id)
            if not start_node or not end_node:
                continue

            output_graph.relationships.append(
                Relationship(
                    source=start_node,
                    target=end_node,
                    type=rel_type,
                ),
            )

        return output_graph
