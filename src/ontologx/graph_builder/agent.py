"""Parser module for parsing log events and constructing knowledge graphs."""

import logging
from dataclasses import dataclass, field, replace
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import ValidationError
from rdflib import Graph

from ontologx.graph_builder.models import BaseEventGraph, build_dynamic_model

logger = logging.getLogger("rich")

INITIALIZE_STATE_NODE = "initialize_state"
BUILD_GRAPH_NODE = "build_graph"
VALIDATE_GRAPH_NODE = "validate_graph"
GET_CORRECTIONS_NODE = "get_corrections"
CLEANUP_MESSAGES_NODE = "cleanup_messages"


@dataclass(frozen=True)
class GraphBuilderInputState:
    """Input state for the graph builder agent."""

    # Original input event text
    input_event: str

    # Examples provided to the LLM for few-shot learning
    input_examples: list[Graph]

    # LLM messages history, including previous events parsed
    messages: list[AnyMessage] = field(default_factory=list)


@dataclass(frozen=True)
class GraphBuilderOutputState:
    """Output state for the graph builder agent."""

    # Parsed and validated knowledge graph
    output_graph: Graph | None = None


@dataclass(frozen=True)
class _GraphBuilderState(GraphBuilderOutputState, GraphBuilderInputState):
    """Full state of the parser agent, with internal only fields."""

    # Number of corrections done so far
    n_corrections_done: int = 0

    # Error encountered when parsing the generated graph
    parsing_error: ValidationError | None = None


@dataclass(frozen=True)
class GraphBuilderContext:
    """Context for the graph builder agent."""

    # LLM used for parsing
    llm: BaseChatModel

    # Ontology used for parsing
    ontology: Graph

    # The prompt used to build the graph
    prompt_build_graph: str

    # Maximum number of conversations to keep in history
    max_conversation_history: int = 5

    # Maximum number of correction steps allowed
    max_correction_steps: int = 3


def _initialize_state(state: _GraphBuilderState, runtime: Runtime[GraphBuilderContext]) -> _GraphBuilderState:
    """Initialize the parser agent state."""
    logger.info("Initializing parser agent state.")

    messages: list[AnyMessage] = (
        state.messages if state.messages else [SystemMessage(runtime.context.prompt_build_graph)]
    )

    # If the last message is not from the human,
    # it means that we are parsing the event for the first time.
    if not isinstance(messages[-1], HumanMessage):
        messages.append(HumanMessage(f"Event: '{state.input_event}'"))

    return _GraphBuilderState(**state.__dict__, messages=messages)


def _build_graph(state: _GraphBuilderState, runtime: Runtime[GraphBuilderContext]) -> _GraphBuilderState:
    """Build the knowledge graph from the input event using the LLM."""
    logger.info("Building knowledge graph for event: %s", {state.input_event})

    try:
        runtime.context.llm.with_structured_output(BaseEventGraph)
    except NotImplementedError as e:
        msg = "The parser model must support structured output."
        raise ValueError(msg) from e

    # Add the graph structure to the structured output.
    # Also include raw output to retrieve eventual errors.
    llm: BaseChatModel = state["llm"].with_structured_output(  # type: ignore[attr-defined]
        build_dynamic_model(runtime.context.ontology),
        include_raw=True,
        method="function_calling",
    )

    raw_output = cast("dict", llm.invoke(state.messages))
    is_output_valid = raw_output.get("parsed", False) and isinstance(raw_output.get("parsed"), BaseEventGraph)

    # If the LLM output is a valid graph we are done.
    if is_output_valid:
        logger.debug("LLM output is a valid graph.")

        return replace(state, output_graph=raw_output.get("parsed"), parsing_error=None)

    logger.debug("LLM output is NOT a valid graph.")

    return replace(
        state,
        output_graph=None,
        parsing_error=cast("ValidationError", raw_output.get("parsing_error"))
        if raw_output.get("parsing_error")
        else None,
    )


def _get_corrections(state: _GraphBuilderState) -> _GraphBuilderState:
    """Generate corrections to the LLM output graph."""
    logger.debug("LLM output invalid. Checking for corrections.")

    correction_msg = "Your answer does not respect the expected ontology or format. Please try again."

    # If there are parsing errors, use them as corrections
    if state.parsing_error and getattr(state.parsing_error, "errors", None):
        errors = [
            {
                "location": ".".join(map(str, err.get("loc"))),
                "invalid_input": err.get("input"),
            }
            for err in state.parsing_error.errors()
        ]

        logger.debug("Parsing errors found: %s", errors)
        correction_msg += f" Fix these errors, without changing anything else: {errors}"

    return replace(
        state,
        n_corrections_done=state.n_corrections_done + 1,
        messages=[*state.messages, HumanMessage(correction_msg)],
    )


def _cleanup_messages(state: _GraphBuilderState, runtime: Runtime[GraphBuilderContext]) -> _GraphBuilderState:
    """Clean up the message history to save memory."""
    logger.info("Cleaning up message history to save memory.")

    # Get the index of the last human event generation message
    last_human_index = next(
        (
            i
            for i in reversed(range(len(state.messages)))
            if (isinstance(state.messages[i], HumanMessage)) and str(state.messages[i].content).startswith("Event: ")
        ),
        -1,
    )

    # Get the index of the last AI message before final tool calls
    last_tool_index = next(
        (i for i in reversed(range(len(state.messages))) if isinstance(state.messages[i], ToolMessage)),
        -1,
    )
    # We only have one tool (structured output), so the last AI message is the one before it
    last_ai_index = last_tool_index - 1 if last_tool_index > 0 else -1

    # Cut off all correction messages (if any),
    # keeping only the initial event generation message and the final AI response.
    # Note: because we use structured output with function calling,
    # the last three messages are always the final AIMessage with the graph,
    # a ToolMessage, and a final AIMessage.
    messages_to_keep = [*state.messages[: last_human_index + 1], state.messages[last_ai_index:]]

    # Find the total number of conversations in history
    n_conversations = sum(
        1
        for i in range(len(messages_to_keep))
        if isinstance(messages_to_keep[i], HumanMessage) and str(messages_to_keep[i].content).startswith("Event: ")
    )

    # If we have more than MAX_CONVERSATION_HISTORY conversations, trim the history
    if n_conversations >= runtime.context.max_conversation_history:
        # Keep only the last 5 conversations
        current_conversations = 0
        for i in reversed(range(len(messages_to_keep))):
            if isinstance(messages_to_keep[i], HumanMessage) and str(messages_to_keep[i].content).startswith("Event: "):
                current_conversations += 1
                if current_conversations == runtime.context.max_conversation_history:
                    messages_to_keep = messages_to_keep[i:]
                    break

    return replace(state, messages=messages_to_keep)


graph_builder_agent = StateGraph(
    _GraphBuilderState,
    context_schema=GraphBuilderContext,
)

graph_builder_agent.add_node(INITIALIZE_STATE_NODE, _initialize_state)
graph_builder_agent.add_node(BUILD_GRAPH_NODE, _build_graph)
graph_builder_agent.add_node(GET_CORRECTIONS_NODE, _get_corrections)
graph_builder_agent.add_node(CLEANUP_MESSAGES_NODE, _cleanup_messages)


def _get_next_node(state: _GraphBuilderState, runtime: Runtime[GraphBuilderContext]) -> str:
    """Determine the next node based on the current state."""
    if state.output_graph is None and state.n_corrections_done < runtime.context.max_correction_steps:
        return GET_CORRECTIONS_NODE

    return CLEANUP_MESSAGES_NODE


graph_builder_agent.add_edge(START, INITIALIZE_STATE_NODE)
graph_builder_agent.add_edge(INITIALIZE_STATE_NODE, BUILD_GRAPH_NODE)
graph_builder_agent.add_conditional_edges(
    BUILD_GRAPH_NODE,
    _get_next_node,
    [CLEANUP_MESSAGES_NODE, GET_CORRECTIONS_NODE],  # This is needed for proper graph visualization
)
graph_builder_agent.add_edge(GET_CORRECTIONS_NODE, BUILD_GRAPH_NODE)
graph_builder_agent.add_edge(CLEANUP_MESSAGES_NODE, END)

memory = MemorySaver()
graph_builder_agent = graph_builder_agent.compile(checkpointer=memory)
