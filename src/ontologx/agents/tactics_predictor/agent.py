"""Tactics predictor agent for analyzing log event graphs and predicting MITRE ATT&CK tactics."""

import logging
from dataclasses import dataclass
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from mitreattack.stix20.MitreAttackData import Tactic, Technique
from rdflib import Graph

from ontologx.agents.tactics_predictor.models import SessionTTPs

logger = logging.getLogger("rich")

PREDICT_TACTICS_NODE = "predict_tactics"

_PROMPT = """
You are a cybersecurity analyst AI. You are given as input a set of knowledge graphs \
representing a log events captured by a honeypot. \
Each knowledge graph encodes entities (e.g., processes, IP addresses, \
files, commands) and their relationships, and all graphs belong to the same session of activity, \
where some form of reconnaissance or attack may have taken place. \
Only logs with event ID "cowrie.command.input" are attacker's commands.
All the other event IDs indicate logs that are not visible to the attacker, \
such as client version, file upload or download, or meta-information about the connection itself. \
Do not confuse these with the attacker's commands. \
Your task is to analyze the combined activity across all these knowledge graphs \
and map them to MITRE ATT&CK enterprise tactics.

You are provided with a tool that allows you to look up MITRE ATT&CK tactics and techniques.
You may use this tool to find relevant tactics based on observed behaviors in the session.

# Rules
You MUST adhere to the following constraints at all times:
1. The output tactics must be matched to the observed behaviors in the session.
2. If multiple tactics apply to the session, include only the ones that you are confident about.
3. The output tactics must be defined in MITRE ATT&CK enterprise.

# Strict Compliance
Adhere to these rules strictly. Any deviation will result in termination.
"""


@dataclass(frozen=True)
class TacticsPredictorInputState:
    """Input state for the graph builder agent."""

    chunk: list[Graph]


@dataclass(frozen=True)
class TacticsPredictorOutputState:
    """Output state for the graph builder agent."""

    # Parsed and validated knowledge graph
    tactics: list[Tactic]

    techniques: list[Technique]


@dataclass(frozen=True)
class _GraphBuilderState(TacticsPredictorOutputState, TacticsPredictorInputState):
    """Full state of the parser agent, with internal only fields."""


@dataclass(frozen=True)
class TacticsPredictorContext:
    """Context for the graph builder agent."""

    # LLM used for parsing
    llm: BaseChatModel


def _predict_tactics(
    state: TacticsPredictorInputState,
    runtime: Runtime[TacticsPredictorContext],
) -> TacticsPredictorOutputState:
    """Predict tactics from the input graphs using the LLM."""
    logger.info("Predicting tactics from input graphs.")

    # Combine all input graphs into a single graph for analysis
    combined_graph = Graph()
    for g in state.chunk:
        combined_graph += g

    # Prepare the prompt with relevant information extracted from the graph
    prompt = "Analyze the following log events and predict the MITRE ATT&CK tactics:\n"
    for s, p, o in combined_graph:
        prompt += f"{s} {p} {o}\n"

    # Invoke the LLM to get tactics
    llm = runtime.context.llm.with_structured_output(SessionTTPs)
    response = llm.invoke([SystemMessage(_PROMPT), HumanMessage(prompt)])
    response = cast("SessionTTPs", response)

    return TacticsPredictorOutputState(tactics=response.tactics, techniques=response.techniques)


tactics_predictor_agent = StateGraph(
    _GraphBuilderState,
    context_schema=TacticsPredictorContext,
    input_schema=TacticsPredictorInputState,
    output_schema=TacticsPredictorOutputState,
)

tactics_predictor_agent.add_node(PREDICT_TACTICS_NODE, _predict_tactics)


tactics_predictor_agent.add_edge(START, PREDICT_TACTICS_NODE)
tactics_predictor_agent.add_edge(PREDICT_TACTICS_NODE, END)

tactics_predictor_agent = tactics_predictor_agent.compile()
