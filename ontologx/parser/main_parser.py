"""Parser module for parsing log events and constructing knowledge graphs."""

import logging
import uuid
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j.graphs.graph_document import GraphDocument
from pydantic import ValidationError

from ontologx.parser.models import BaseEventGraph, build_dynamic_model
from ontologx.parser.parser import Parser
from ontologx.parser.tools import fetch_ip_address_info
from ontologx.store import Store

logger = logging.getLogger("rich")


def _example_message_group(event: str, graph: GraphDocument, context: dict) -> list[BaseMessage]:
    """Create an example message group for the given event and graph."""
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

    tool_call_id = f"call_{uuid.uuid4()!s}"

    return [
        HumanMessage(f"Event: '{event}'\nContext: {context}", name="example_user"),
        AIMessage(
            "",
            id=f"run_{uuid.uuid4()!s}",
            tool_calls=[
                {
                    "name": "EventGraph",  # This must be the name of the class returned by structured output
                    "args": {
                        "nodes": nodes,
                        "relationships": relationships,
                    },
                    "id": tool_call_id,
                },
            ],
        ),
        ToolMessage("", tool_call_id=tool_call_id),
        AIMessage("Done"),
    ]


class MainParser(Parser):
    """Parser class for parsing log events and constructing knowledge graphs using a LLM."""

    def __init__(
        self,
        llm: BaseChatModel,
        store: Store,
        prompt_build_graph: str,
        correction_steps: int,
    ) -> None:
        super().__init__(llm, store, prompt_build_graph)
        self.correction_steps = correction_steps

        try:
            llm.with_structured_output(BaseEventGraph)
        except NotImplementedError as e:
            msg = "The parser model must support structured output."
            raise ValueError(msg) from e

        # Add context enrichment tools.
        # Note: not all models support tools + structured output
        structured_model = llm.bind_tools([fetch_ip_address_info])

        # Add the graph structure to the structured output.
        # Also include raw output to retrieve eventual errors.
        structured_model = structured_model.with_structured_output(  # type: ignore[attr-defined]
            build_dynamic_model(store.ontology.graph()),
            include_raw=True,
        )

        gen_graph_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_build_graph),
                ("placeholder", "{examples}"),
                ("human", "Event: '{event}'\nContext: '{context}'"),
                ("placeholder", "{corrections}"),
            ],
        )

        self.chain = gen_graph_prompt | structured_model

    def __get_examples(self, event: str) -> list[BaseMessage]:
        similar_events = self.store.dataset.events_mmr_search(event, k=2)

        messages = []
        for similar_event, graph in similar_events:
            source_node = next((node for node in graph.nodes if node.type == "Source"), None)

            context = {}
            if source_node:
                context["source"] = source_node.properties.get("sourceName", "")
                context["device"] = source_node.properties.get("sourceDevice", "")

            messages.extend(_example_message_group(similar_event, graph, context))

        return messages

    def parse(self, event: str, context: dict) -> GraphDocument | None:
        """Parse the given event and construct a knowledge graph.

        Args:
            event: The log event to parse.
            context: The context of the event.

        Returns:
            A report containing the stats of the parsing process.

        """
        # Retrieve examples once for all the self-reflection steps
        examples = self.__get_examples(event)

        corrections = []

        # Using self_reflection_steps + 1 to account for the initial attempt
        for current_step in range(self.correction_steps + 1):
            logger.debug("Correction step %d", current_step)

            raw_schema = self.chain.invoke(
                {
                    "event": event,
                    "context": context,
                    "examples": examples,
                    "corrections": corrections,
                },
            )

            raw_schema = cast(dict, raw_schema)

            # Error handling for when the output is not parsed correctly
            if not raw_schema.get("parsed"):
                logger.debug("LLM output not parsed correctly. Checking for corrections.")

                try:
                    llm_answer = cast(AIMessage, raw_schema["raw"])
                    # Create a new AIMessage with the same content and tool_calls,
                    # but without all the unnecessary stuff
                    corrections.append(
                        AIMessage(llm_answer.content, id=llm_answer.id, tool_calls=llm_answer.tool_calls),
                    )
                except KeyError:
                    logger.debug("No raw LLM output found.")

                    # If the LLM gives no output, retry again with no corrections
                    continue

                msg = "Your answer does not respect the expected format. Please try again."

                # If there are parsing errors, use them as corrections
                if raw_schema.get("parsing_error"):
                    parsing_error = cast(ValidationError, raw_schema["parsing_error"])
                    errors = [
                        {
                            "location": ".".join(map(str, err.get("loc"))),
                            "invalid_input": err.get("input"),
                        }
                        for err in parsing_error.errors()
                    ]

                    logger.debug("Parsing errors found: %s", errors)
                    msg += f" Fix these errors, without modifying anything else: {errors}"

                corrections.append(HumanMessage(msg))

                continue

            output_graph: GraphDocument = raw_schema["parsed"].graph()

            logger.debug("Graph constructed successfully.")
            return output_graph

        return None
