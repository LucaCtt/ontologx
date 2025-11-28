"""Parser module for parsing log events and constructing knowledge graphs."""

import logging
from typing import TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import ValidationError

from ontologx.graph_builder.models import BaseEventGraph, build_dynamic_model
from ontologx.store import GraphDocument

logger = logging.getLogger("rich")

INITIALIZE_STATE_NODE = "initialize_state"
BUILD_GRAPH_NODE = "build_graph"
VALIDATE_GRAPH_NODE = "validate_graph"
GET_CORRECTIONS_NODE = "get_corrections"
ERROR_NODE = "error"
CLEANUP_HISTORY_NODE = "cleanup_history"

MAX_CONVERSATION_HISTORY = 5


class GraphBuilderState(TypedDict):
    """State of the parser agent."""

    # Original input event text
    input_event: str

    # Context of the input event
    input_context: dict

    # Examples provided to the LLM for few-shot learning
    input_examples: list[GraphDocument]

    # LLM used for parsing
    llm: BaseChatModel

    # Ontology used for parsing
    ontology: GraphDocument

    # The prompt used to build the graph
    prompt_build_graph: str

    # Parsed and validated knowledge graph
    output_graph: GraphDocument | None

    # Maximum number of correction steps allowed
    max_correction_steps: int

    # Number of corrections done so far
    n_corrections_done: int

    # Error encountered when parsing the generated graph
    parsing_error: ValidationError | None

    # LLM messages history, including previous events parsed
    messages: list[AnyMessage]


def initialize_state(state: GraphBuilderState) -> Command:
    """Initialize the parser agent state."""
    logger.info("Initializing parser agent state.")

    messages: list[AnyMessage] = (
        state["messages"] if state.get("messages") else [SystemMessage(state["prompt_build_graph"])]
    )

    # If the last message is not from the human,
    # it means that we are parsing the event for the first time.
    if not isinstance(messages[-1], HumanMessage):
        messages.append(HumanMessage(f"Event: '{state['input_event']}', Context: {state['input_context']}"))

    try:
        state["llm"].with_structured_output(BaseEventGraph)
    except NotImplementedError as e:
        msg = "The parser model must support structured output."
        raise ValueError(msg) from e

    # Add the graph structure to the structured output.
    # Also include raw output to retrieve eventual errors.
    structured_llm = state["llm"].with_structured_output(  # type: ignore[attr-defined]
        build_dynamic_model(state["ontology"]),
        include_raw=True,
        method="function_calling",
    )

    return Command(
        update={
            "output_graph": None,
            "n_corrections_done": 0,
            "parsing_errors": [],
            "messages": messages,
            "llm": structured_llm,
        },
        goto=BUILD_GRAPH_NODE,
    )


def build_graph(state: GraphBuilderState) -> Command:
    """Build the knowledge graph from the input event using the LLM."""
    logger.info("Building knowledge graph for event: %s", {state["input_event"]})

    raw_output = cast("dict", state["llm"].invoke(state["messages"]))
    is_output_valid = raw_output.get("parsed", False) and isinstance(raw_output.get("parsed"), BaseEventGraph)

    # If the LLM output is a valid graph we are done.
    if is_output_valid:
        logger.debug("LLM output is a valid graph.")

        return Command(
            update={"output_graph": raw_output.get("parsed")},
            goto=CLEANUP_HISTORY_NODE,
        )

    # If the output is invalid, check if we can do corrections.
    if state["n_corrections_done"] < state["max_correction_steps"]:
        logger.debug("LLM output invalid, %d correction steps remaining.")

        parsing_error = (
            cast("ValidationError", raw_output.get("parsing_error")) if raw_output.get("parsing_error") else None
        )

        return Command(
            update={"parsing_error": parsing_error},
            goto=GET_CORRECTIONS_NODE,
        )

    # Otherwise, go to error node.
    logger.debug("LLM output invalid and no correction steps remaining.")

    return Command(
        goto=ERROR_NODE,
    )


def correct_graph(state: GraphBuilderState) -> Command:
    """Generate corrections to the LLM output graph."""
    logger.debug("LLM output invalid. Checking for corrections.")

    correction_msg = "Your answer does not respect the expected ontology or format. Please try again."

    # If there are parsing errors, use them as corrections
    if state["parsing_error"] and getattr(state["parsing_error"], "errors", None):
        errors = [
            {
                "location": ".".join(map(str, err.get("loc"))),
                "invalid_input": err.get("input"),
            }
            for err in state["parsing_error"].errors()
        ]

        logger.debug("Parsing errors found: %s", errors)
        correction_msg += f" Fix these errors, without changing anything else: {errors}"

    return Command(
        update={
            "n_corrections_done": state["n_corrections_done"] + 1,
            "messages": [*state["messages"], HumanMessage(correction_msg)],
        },
        goto=BUILD_GRAPH_NODE,
    )


def error(state: GraphBuilderState) -> Command:
    """Handle the error when the graph cannot be parsed after corrections."""
    logger.info("Unable to parse the event into a valid graph after %d attempts.", state["max_correction_steps"])

    return Command(
        update={
            "output_graph": None,
            "messages": [*state["messages"][:-1], AIMessage("Unable to parse the event into a valid graph.")],
        },
        goto=CLEANUP_HISTORY_NODE,
    )


def cleanup_history(state: GraphBuilderState) -> Command:
    """Clean up the message history to save memory."""
    logger.info("Cleaning up message history to save memory.")

    # Get the index of the last human event generation message
    last_human_index = next(
        (
            i
            for i in reversed(range(len(state["messages"])))
            if (isinstance(state["messages"][i], HumanMessage))
            and str(state["messages"][i].content).startswith("Event: ")
        ),
        -1,
    )

    # Get the index of the last AI message before final tool calls
    last_tool_index = next(
        (i for i in reversed(range(len(state["messages"]))) if isinstance(state["messages"][i], ToolMessage)),
        -1,
    )
    # We only have one tool (structured output), so the last AI message is the one before it
    last_ai_index = last_tool_index - 1 if last_tool_index > 0 else -1

    # Cut off all correction messages (if any),
    # keeping only the initial event generation message and the final AI response.
    # Note: because we use structured output with function calling,
    # the last three messages are always the final AIMessage with the graph,
    # a ToolMessage, and a final AIMessage.
    messages_to_keep = [*state["messages"][: last_human_index + 1], state["messages"][last_ai_index:]]

    # Find the total number of conversations in history
    n_conversations = sum(
        1
        for i in range(len(messages_to_keep))
        if isinstance(messages_to_keep[i], HumanMessage) and str(messages_to_keep[i].content).startswith("Event: ")
    )

    # If we have more than MAX_CONVERSATION_HISTORY conversations, trim the history
    if n_conversations >= MAX_CONVERSATION_HISTORY:
        # Keep only the last 5 conversations
        current_conversations = 0
        for i in reversed(range(len(messages_to_keep))):
            if isinstance(messages_to_keep[i], HumanMessage) and str(messages_to_keep[i].content).startswith("Event: "):
                current_conversations += 1
                if current_conversations == MAX_CONVERSATION_HISTORY:
                    messages_to_keep = messages_to_keep[i:]
                    break

    return Command(
        update={
            "messages": messages_to_keep,
        },
        goto=END,
    )


graph_builder_agent = StateGraph(GraphBuilderState)

graph_builder_agent.add_node(INITIALIZE_STATE_NODE, initialize_state)
graph_builder_agent.add_node(BUILD_GRAPH_NODE, build_graph)
graph_builder_agent.add_node(GET_CORRECTIONS_NODE, correct_graph)
graph_builder_agent.add_node(ERROR_NODE, error)
graph_builder_agent.add_node(CLEANUP_HISTORY_NODE, cleanup_history)

graph_builder_agent.add_edge(START, INITIALIZE_STATE_NODE)

memory = MemorySaver()
graph_builder_agent = graph_builder_agent.compile(checkpointer=memory)
