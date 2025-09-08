"""Module to calculate metrics for Tactics, Techniques, and Procedures (TTPs) predictions."""

import functools
from enum import StrEnum
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ontologx.store import GraphDocument


class _MITRETactic(StrEnum):
    """Enumeration of MITRE ATT&CK tactics."""

    RECONNAISSANCE = "Reconnaissance"
    RESOURCE_DEVELOPMENT = "Resource Development"
    INITIAL_ACCESS = "Initial Access"
    EXECUTION = "Execution"
    PERSISTENCE = "Persistence"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    DEFENSE_EVASION = "Defense Evasion"
    CREDENTIAL_ACCESS = "Credential Access"
    DISCOVERY = "Discovery"
    LATERAL_MOVEMENT = "Lateral Movement"
    COLLECTION = "Collection"
    COMMAND_AND_CONTROL = "Command and Control"
    EXFILTRATION = "Exfiltration"
    IMPACT = "Impact"


class _SessionTactics(BaseModel):
    """List of MITRE ATT&CK tactics observed in a session."""

    tactics: list[_MITRETactic] = Field(
        description="List of MITRE ATT&CK tactics observed in the session.",
        examples=[_MITRETactic.INITIAL_ACCESS, _MITRETactic.EXECUTION, _MITRETactic.PERSISTENCE],
    )


class _TacticsPredictor:
    def __init__(self, llm: BaseChatModel, prompt_predict_tactics: str):
        # Check if the model supports structured output
        try:
            model = llm.with_structured_output(_SessionTactics, method="function_calling")
        except NotImplementedError as e:
            msg = "The predictor model must support structured output."
            raise ValueError(msg) from e

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_predict_tactics),
                ("human", "Session: '{session}'"),
            ],
        )

        self.__chain = prompt | model

    def predict_tactics(self, graphs: list[GraphDocument]) -> list[_MITRETactic]:
        """Predict the MITRE ATT&CK tactics for a given session of logs given as graphs.

        Args:
            graphs (list[GraphDocument]): A list of graph documents representing the session.

        Returns:
            list[MITRETactic]: A list of predicted MITRE ATT&CK tactics.

        """
        out = self.__chain.invoke({"session": graphs})

        return cast("_SessionTactics", out).tactics


class _SessionTacticMetrics:
    def __init__(self, y_labels_pred: list[_MITRETactic], y_labels_true: list[_MITRETactic]):
        predicted_set = set(y_labels_pred)
        ground_truth_set = set(y_labels_true)

        self.__tp = len(predicted_set & ground_truth_set)
        self.__fp = len(predicted_set - ground_truth_set)  # in predicted but not in ground truth
        self.__fn = len(ground_truth_set - predicted_set)  # in ground truth but not in predicted

    @functools.cached_property
    def precision(self) -> float:
        """Calculate the precision of the predicted tactics.

        Returns:
            float: Precision of the predicted tactics.

        """
        return self.__tp / (self.__tp + self.__fp) if (self.__tp + self.__fp) > 0 else 0.0

    @functools.cached_property
    def recall(self) -> float:
        """Calculate the recall of the predicted tactics.

        Returns:
            float: Recall of the predicted tactics.

        """
        return self.__tp / (self.__tp + self.__fn) if (self.__tp + self.__fn) > 0 else 0.0

    @functools.cached_property
    def f1_score(self) -> float:
        """Calculate the F1 score of the predicted tactics.

        Returns:
            float: F1 score of the predicted tactics.

        """
        precision = self.precision
        recall = self.recall
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


class TacticsMetrics:
    """Class to calculate metrics for MITRE Tactics predictions."""

    def __init__(
        self,
        y_pred: list[GraphDocument],
        y_true: list[GraphDocument],
        llm: BaseChatModel,
        prompt_predict_tactics: str,
    ):
        sessions_dict = {}
        tactics_dict = {}
        for pred, true in zip(y_pred, y_true, strict=True):
            event_node = next(e for e in true.nodes if e.type == "olx:Event")
            session_id = event_node.properties.get("olx:eventSessionID")

            if session_id is None:
                continue

            if session_id not in sessions_dict:
                sessions_dict[session_id] = []
                tactics_dict[session_id] = []

            sessions_dict[session_id].append(pred)
            true_tactics = [_MITRETactic(i) for i in true.source.metadata["tactics"]]
            tactics_dict[session_id] = list(set(tactics_dict[session_id] + true_tactics))

        self.__y_pred_sessions = [sessions_dict[session_id] for session_id in sessions_dict]
        self.__y_true_tactics = [tactics_dict[session_id] for session_id in sessions_dict]
        self.__llm = llm
        self.__prompt_predict_tactics = prompt_predict_tactics

    def precision(self) -> float:
        """Calculate the overall precision of the predicted tactics across all sessions."""
        return (
            sum(m.precision for m in self.__session_metrics) / len(self.__session_metrics)
            if self.__session_metrics
            else 0.0
        )

    def recall(self) -> float:
        """Calculate the overall recall of the predicted tactics across all sessions."""
        return (
            sum(m.recall for m in self.__session_metrics) / len(self.__session_metrics)
            if self.__session_metrics
            else 0.0
        )

    def f1_score(self) -> float:
        """Calculate the overall F1 score of the predicted tactics across all sessions."""
        return (
            sum(m.f1_score for m in self.__session_metrics) / len(self.__session_metrics)
            if self.__session_metrics
            else 0.0
        )

    @functools.cached_property
    def __session_metrics(self) -> list[_SessionTacticMetrics]:
        """Predict tactics for each session using the predictor."""
        predictor = _TacticsPredictor(
            llm=self.__llm,
            prompt_predict_tactics=self.__prompt_predict_tactics,
        )
        pred_tactics = [predictor.predict_tactics(session) for session in self.__y_pred_sessions]
        return [
            _SessionTacticMetrics(y_pred, y_true)
            for y_pred, y_true in zip(pred_tactics, self.__y_true_tactics, strict=True)
        ]
