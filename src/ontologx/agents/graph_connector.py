"""Graph Connector agent for processing and merging graphs."""

from dataclasses import dataclass, replace

import polars as pl
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from rdflib import Graph

from ontologx.agents.graph_builder.agent import GraphBuilderContext, GraphBuilderInputState, graph_builder_agent
from ontologx.agents.tactics_predictor.agent import (
    TacticsPredictorContext,
    TacticsPredictorInput,
    tactics_predictor_agent,
)
from ontologx.stores import GraphStore, VectorStore

CHUNK_GRAPHS_NODE = "chunk_graphs"
BUILD_GRAPHS_FOR_CHUNK_NODE = "build_graphs_for_chunk"
PREDICT_TTPS_FOR_CHUNK_NODE = "predict_ttps"
MERGE_GRAPHS_NODE = "merge_graphs"


@dataclass(frozen=True)
class GraphConnectorInput:
    """Input for the Graph Connector agent."""

    # The list of graphs to process
    events: pl.DataFrame


@dataclass(frozen=True)
class _GraphConnectorState(GraphConnectorInput):
    chunks: list[pl.DataFrame]

    current_chunk_index = 0


@dataclass(frozen=True)
class GraphConnectorContext:
    """Context for the Graph Connector agent."""

    # The LLM to use for TTP prediction
    llm: BaseChatModel

    # The ontology for the log graphs
    ontology: Graph

    # The vector store to save embeddings
    vector_store: VectorStore

    # The graph store to save the merged graphs
    graph_store: GraphStore

    # Maximum number of graph conversation history to keep in memory
    max_graph_conversation_history: int = 5

    # Maximum number of graph correction steps
    max_graph_correction_steps: int = 3


def _chunk_graphs(state: GraphConnectorInput) -> _GraphConnectorState:
    # Group by application and device
    chunks = [df for _, df in state.events.group_by(["application", "device"])]

    return _GraphConnectorState(
        **state.__dict__,
        chunks=chunks,
    )


def _build_graphs_for_chunk(
    state: _GraphConnectorState,
    runtime: Runtime[GraphConnectorContext],
) -> _GraphConnectorState:
    # Build a graph for the current chunk of events
    current_chunk = state.chunks[state.current_chunk_index]

    # Reset memory saver for graph builder agent
    graph_builder_agent.checkpointer = MemorySaver()

    for row in current_chunk.iter_rows(named=True):
        relevant_events = runtime.context.vector_store.search(row["event_text"])
        relevant_kgs = [runtime.context.graph_store.get_graph(event) for event in relevant_events]

        output = graph_builder_agent.invoke(
            GraphBuilderInputState(
                input_event=row["event_text"],
                input_examples=relevant_kgs,
            ),
            context=GraphBuilderContext(
                llm=runtime.context.llm,
                ontology=runtime.context.ontology,
                max_conversation_history=runtime.context.max_graph_conversation_history,
                max_correction_steps=runtime.context.max_graph_correction_steps,
            ),
        )
        row["graph"] = output["output_graph"]

    # Maybe inefficient
    state.chunks[state.current_chunk_index] = current_chunk
    return replace(state, chunks=state.chunks)


def _predict_ttps_for_chunk(
    state: _GraphConnectorState,
    runtime: Runtime[GraphConnectorContext],
) -> _GraphConnectorState:
    current_chunk = state.chunks[state.current_chunk_index]

    # Predict TTPs for chunk
    output = tactics_predictor_agent.invoke(
        TacticsPredictorInput(chunk=current_chunk["graph"].to_list()),
        context=TacticsPredictorContext(llm=runtime.context.llm),
    )

    # Assign TTPs to the current chunk
    current_chunk["tactics"] = [output["tactics"]] * len(current_chunk)
    current_chunk["techniques"] = [output["techniques"]] * len(current_chunk)

    state.chunks[state.current_chunk_index] = current_chunk
    return replace(state, chunks=state.chunks, current_chunk_index=state.current_chunk_index + 1)


def _merge_graphs(state: _GraphConnectorState) -> _GraphConnectorState:
    # Merge

    return state


graph_connector_agent = StateGraph(
    _GraphConnectorState,
    input_schema=GraphConnectorInput,
    context_schema=GraphConnectorContext,
)


def _get_next_node(state: _GraphConnectorState) -> str:
    if state.current_chunk_index >= len(state.chunks):
        return MERGE_GRAPHS_NODE

    return BUILD_GRAPHS_FOR_CHUNK_NODE


graph_connector_agent.add_node(CHUNK_GRAPHS_NODE, _chunk_graphs)
graph_connector_agent.add_node(BUILD_GRAPHS_FOR_CHUNK_NODE, _build_graphs_for_chunk)
graph_connector_agent.add_node(PREDICT_TTPS_FOR_CHUNK_NODE, _predict_ttps_for_chunk)
graph_connector_agent.add_node(MERGE_GRAPHS_NODE, _merge_graphs)

graph_connector_agent.add_edge(START, CHUNK_GRAPHS_NODE)
graph_connector_agent.add_edge(CHUNK_GRAPHS_NODE, PREDICT_TTPS_FOR_CHUNK_NODE)
graph_connector_agent.add_edge(PREDICT_TTPS_FOR_CHUNK_NODE, BUILD_GRAPHS_FOR_CHUNK_NODE)
graph_connector_agent.add_conditional_edges(
    BUILD_GRAPHS_FOR_CHUNK_NODE,
    _get_next_node,
    [MERGE_GRAPHS_NODE, BUILD_GRAPHS_FOR_CHUNK_NODE],
)
graph_connector_agent.add_edge(MERGE_GRAPHS_NODE, END)


memory = MemorySaver()
graph_connector_agent = graph_connector_agent.compile(checkpointer=memory)
