"""Module to calculate metrics for Tactics, Techniques, and Procedures (TTPs) predictions."""

import functools
from enum import StrEnum
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ontologx.store import GraphDocument


class MITRETactic(StrEnum):
    """Enumeration of MITRE ATT&CK enterprise tactics."""

    EXECUTION = "Execution"
    DISCOVERY = "Discovery"
    INITIAL_ACCESS = "Initial Access"
    PERSISTENCE = "Persistence"
    EXFILTRATION = "Exfiltration"
    IMPACT = "Impact"
    DEFENSE_EVASION = "Defense Evasion"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    CREDENTIAL_ACCESS = "Credential Access"
    LATERAL_MOVEMENT = "Lateral Movement"
    COLLECTION = "Collection"
    COMMAND_AND_CONTROL = "Command and Control"


class _SessionTactics(BaseModel):
    """List of MITRE ATT&CK tactics observed in a session."""

    tactics: list[MITRETactic] = Field(
        description="List of MITRE ATT&CK tactics observed in the session.",
        examples=[MITRETactic.INITIAL_ACCESS, MITRETactic.EXECUTION, MITRETactic.PERSISTENCE],
    )


class TacticsPredictor:
    """Class to predict MITRE ATT&CK tactics for a given session of logs."""

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

    def predict_tactics(self, graphs: list[GraphDocument]) -> list[MITRETactic]:
        """Predict the MITRE ATT&CK tactics for a given session of logs given as graphs.

        Args:
            graphs (list[GraphDocument]): A list of graph documents representing the session.

        Returns:
            list[MITRETactic]: A list of predicted MITRE ATT&CK tactics.

        """
        out = self.__chain.invoke({"session": graphs})

        return cast("_SessionTactics", out).tactics


class SessionTacticsMetrics:
    """Class to calculate metrics for MITRE Tactics predictions for a single session."""

    def __init__(self, y_labels_pred: list[MITRETactic], y_labels_true: list[MITRETactic]):
        self.__predicted_set = set(y_labels_pred)
        self.__ground_truth_set = set(y_labels_true)

    @functools.cached_property
    def precisions(self) -> dict[MITRETactic, float]:
        """Calculate the precision of the predicted tactics.

        Returns:
            float: Precision of the predicted tactics.

        """
        precisions = {}

        for tactic in self.__ground_truth_set.union(self.__predicted_set):
            tp = len(self.__predicted_set.intersection(self.__ground_truth_set).intersection({tactic}))
            fp = len(self.__predicted_set.difference(self.__ground_truth_set).intersection({tactic}))
            precisions[tactic] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        return precisions

    @functools.cached_property
    def recalls(self) -> dict[MITRETactic, float]:
        """Calculate the recall of the predicted tactics.

        Returns:
            float: Recall of the predicted tactics.

        """
        recalls = {}

        for tactic in self.__ground_truth_set.union(self.__predicted_set):
            tp = len(self.__predicted_set.intersection(self.__ground_truth_set).intersection({tactic}))
            fn = len(self.__ground_truth_set.difference(self.__predicted_set).intersection({tactic}))
            recalls[tactic] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return recalls

    @functools.cached_property
    def f1_scores(self) -> dict[MITRETactic, float]:
        """Calculate the F1 score of the predicted tactics.

        Returns:
            float: F1 score of the predicted tactics.

        """
        f1_scores = {}
        for tactic in self.__ground_truth_set.union(self.__predicted_set):
            precision = self.precisions[tactic]
            recall = self.recalls[tactic]
            f1_scores[tactic] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1_scores


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
            event_node = next(e for e in pred.nodes if e.type == "olx:Event")
            session_id = event_node.properties.get("olx:eventSessionID")

            if session_id is None:
                continue

            if session_id not in sessions_dict:
                sessions_dict[session_id] = []
                tactics_dict[session_id] = []

            sessions_dict[session_id].append(pred)
            true_tactics = [MITRETactic(i) for i in true.source.metadata["tactics"]]
            tactics_dict[session_id] = list(set(tactics_dict[session_id] + true_tactics))

        self.__y_pred_sessions = [sessions_dict[session_id] for session_id in sessions_dict]
        self.__y_true_tactics = [tactics_dict[session_id] for session_id in sessions_dict]
        self.__llm = llm
        self.__prompt_predict_tactics = prompt_predict_tactics

    @functools.cached_property
    def precisions(self) -> dict[MITRETactic, float]:
        """Calculate the overall precision of the predicted tactics across all sessions."""
        all_precisions = {}
        for m in self.__session_metrics:
            for tactic, precision in m.precisions.items():
                if tactic not in all_precisions:
                    all_precisions[tactic] = []
                all_precisions[tactic].append(precision)

        return {tactic: sum(precisions) / len(precisions) for tactic, precisions in all_precisions.items()}

    @functools.cached_property
    def recalls(self) -> dict[MITRETactic, float]:
        """Calculate the overall recall of the predicted tactics across all sessions."""
        all_recalls = {}
        for m in self.__session_metrics:
            for tactic, recall in m.recalls.items():
                if tactic not in all_recalls:
                    all_recalls[tactic] = []
                all_recalls[tactic].append(recall)

        return {tactic: sum(recalls) / len(recalls) for tactic, recalls in all_recalls.items()}

    @functools.cached_property
    def f1_scores(self) -> dict[MITRETactic, float]:
        """Calculate the overall F1 score of the predicted tactics across all sessions."""
        all_f1_scores = {}
        for m in self.__session_metrics:
            for tactic, f1_score in m.f1_scores.items():
                if tactic not in all_f1_scores:
                    all_f1_scores[tactic] = []
                all_f1_scores[tactic].append(f1_score)

        return {tactic: sum(f1_scores) / len(f1_scores) for tactic, f1_scores in all_f1_scores.items()}

    @functools.cached_property
    def __session_metrics(self) -> list[SessionTacticsMetrics]:
        """Predict tactics for each session using the predictor."""
        predictor = TacticsPredictor(
            llm=self.__llm,
            prompt_predict_tactics=self.__prompt_predict_tactics,
        )
        pred_tactics = [predictor.predict_tactics(session) for session in self.__y_pred_sessions]
        return [
            SessionTacticsMetrics(y_pred, y_true)
            for y_pred, y_true in zip(pred_tactics, self.__y_true_tactics, strict=True)
        ]
