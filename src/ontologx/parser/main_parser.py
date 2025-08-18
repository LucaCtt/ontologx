"""Parser module for parsing log events and constructing knowledge graphs."""

import logging
import uuid
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from ontologx.parser.models import BaseEventGraph, build_dynamic_model
from ontologx.parser.parser import Parser
from ontologx.store import GraphDocument, Store

logger = logging.getLogger("rich")


def _example_message_group(event_graph: GraphDocument) -> list[BaseMessage]:
    """Create an example message group for the given event and graph."""
    event = event_graph.source.page_content
    context = event_graph.source.metadata

    nodes = [
        {
            "id": node.id,
            "type": node.type,
            "properties": [{"type": key, "value": value} for key, value in node.properties.items()],
        }
        for node in event_graph.nodes
    ]

    relationships = [
        {
            "source_id": rel.source.id,
            "target_id": rel.target.id,
            "type": rel.type,
        }
        for rel in event_graph.relationships
    ]

    tool_call_id = str(uuid.uuid4())

    return [
        HumanMessage(f"Event: '{event}'\nContext: {context}", name="example_user"),
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


class MainParser(Parser):
    """Parser class for parsing log events and constructing knowledge graphs using a LLM."""

    def __init__(
        self,
        llm: BaseChatModel,
        store: Store,
        prompt_build_graph: str,
        correction_steps: int,
        examples_retrieval: bool,
    ) -> None:
        super().__init__(llm, store, prompt_build_graph)
        self.correction_steps = correction_steps
        self.examples_retrieval = examples_retrieval

        try:
            llm.with_structured_output(BaseEventGraph)
        except NotImplementedError as e:
            msg = "The parser model must support structured output."
            raise ValueError(msg) from e

        # Add the graph structure to the structured output.
        # Also include raw output to retrieve eventual errors.
        structured_model = llm.with_structured_output(  # type: ignore[attr-defined]
            build_dynamic_model(store.ontology()),
            include_raw=True,
            method="function_calling",
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

    def __get_examples(self, event: str, context: dict) -> list[BaseMessage]:
        similar_events = self.store.search("mmr", event, context, k=3)
        return [msg for similar_event in similar_events for msg in _example_message_group(similar_event)]

    def parse(self, event: str, context: dict) -> GraphDocument | None:
        """Parse the given event and construct a knowledge graph.

        Args:
            event: The log event to parse.
            context: The context of the event.

        Returns:
            A report containing the stats of the parsing process.

        """
        # Retrieve examples once for all the self-reflection steps
        examples = self.__get_examples(event, context) if self.examples_retrieval else []

        corrections = []

        # Using self_reflection_steps + 1 to account for the initial attempt
        for current_step in range(self.correction_steps + 1):
            if current_step > 0:
                logger.debug("Correction step %d", current_step)

            raw_schema = self.chain.invoke(
                {
                    "event": event,
                    "context": context,
                    "examples": examples,
                    "corrections": corrections,
                },
            )

            raw_schema = cast("dict", raw_schema)

            # Error handling for when the output is not parsed correctly
            if not raw_schema.get("parsed"):
                # Short-circuit if no corrections are allowed
                if self.correction_steps == 0:
                    return None

                logger.debug("LLM output invalid. Checking for corrections.")

                try:
                    llm_answer = cast("AIMessage", raw_schema["raw"])
                    # Create a new AIMessage with the same content and tool_calls,
                    # but without all the unnecessary stuff
                    corrections.extend(
                        [
                            AIMessage(llm_answer.content, id=llm_answer.id, tool_calls=llm_answer.tool_calls),
                            *[ToolMessage("", tool_call_id=tool_call["id"]) for tool_call in llm_answer.tool_calls],
                            AIMessage("Done"),
                        ],
                    )

                except KeyError:
                    logger.debug("No raw LLM output found.")

                    # If the LLM gives no output, retry again with no corrections
                    continue

                msg = "Your answer does not respect the expected format. Please try again."

                # If there are parsing errors, use them as corrections
                if raw_schema.get("parsing_error") and getattr(raw_schema["parsing_error"], "errors", None):
                    parsing_error = cast("ValidationError", raw_schema["parsing_error"])
                    errors = [
                        {
                            "location": ".".join(map(str, err.get("loc"))),
                            "invalid_input": err.get("input"),
                        }
                        for err in parsing_error.errors()
                    ]

                    logger.debug("Parsing errors found: %s", errors)
                    msg += f" Fix these errors, without changing anything else: {errors}"

                corrections.append(HumanMessage(msg))

                continue

            output_graph: GraphDocument = raw_schema["parsed"].graph(event, context)

            logger.debug("Graph constructed successfully.")
            return output_graph

        return None
