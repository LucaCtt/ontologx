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
        self.pred_labels = set(y_labels_pred)
        self.true_labels = set(y_labels_true)

    @functools.cached_property
    def precision(self) -> float:
        """Calculate the precision of the predicted tactics.

        Returns:
            float: Precision of the predicted tactics.

        """
        tp = len(self.pred_labels.intersection(self.true_labels))
        fp = len(self.pred_labels.difference(self.true_labels))

        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @functools.cached_property
    def recall(self) -> float:
        """Calculate the recall of the predicted tactics.

        Returns:
            float: Recall of the predicted tactics.

        """
        tp = len(self.pred_labels.intersection(self.true_labels))
        fn = len(self.true_labels.difference(self.pred_labels))

        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @functools.cached_property
    def f1_score(self) -> float:
        """Calculate the F1 score of the predicted tactics.

        Returns:
            float: F1 score of the predicted tactics.

        """
        precision = self.precision
        recall = self.recall

        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def group_events_by_session(graphs: list[GraphDocument]) -> dict[str, list[GraphDocument]]:
    """Group event graphs by their session ID.

    Args:
        graphs (list[GraphDocument]): A list of graph documents representing events.

    Returns:
        dict[str, list[GraphDocument]]: A dictionary mapping session IDs to lists of graph documents.

    """
    sessions = {}
    for graph in graphs:
        event_node = next(e for e in graph.nodes if e.type == "olx:Event")
        session_id = event_node.properties.get("olx:eventSessionID")

        if session_id is None:
            continue

        if session_id not in sessions:
            sessions[session_id] = []

        sessions[session_id].append(graph)

    return sessions


class TacticsMetrics:
    """Class to calculate metrics for MITRE Tactics predictions."""

    def __init__(
        self,
        y_pred_sessions: dict[str, list[GraphDocument]],
        y_true_tactics: dict[str, list[MITRETactic]],
        llm: BaseChatModel,
        prompt_predict_tactics: str,
    ):
        self.__y_pred_sessions = y_pred_sessions
        self.__y_true_tactics = y_true_tactics
        self.__llm = llm
        self.__prompt_predict_tactics = prompt_predict_tactics

    @classmethod
    def from_ungrouped_events(
        cls,
        y_pred: list[GraphDocument],
        y_true: list[GraphDocument],
        llm: BaseChatModel,
        prompt_predict_tactics: str,
    ) -> "TacticsMetrics":
        """Create a TacticsMetrics instance from ungrouped event graphs.

        Args:
            y_pred (list[GraphDocument]): A list of graph documents representing events.
            y_true (dict[str, list[MITRETactic]]): A dictionary mapping session IDs
                                                   to lists of true MITRE ATT&CK tactics.
            llm (BaseChatModel): The language model to use for predictions.
            prompt_predict_tactics (str): The prompt template for predicting tactics.

        Returns:
            TacticsMetrics: An instance of TacticsMetrics.

        """
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
            true_tactics = [MITRETactic(i.lower().title()) for i in true.source.metadata["tactics"]]
            tactics_dict[session_id] = list(set(tactics_dict[session_id] + true_tactics))

        return cls(sessions_dict, tactics_dict, llm, prompt_predict_tactics)

    @functools.cached_property
    def precision(self) -> float:
        """Calculate the overall precision of the predicted tactics across all sessions."""
        return sum(m.precision for m in self.__session_metrics) / len(self.__session_metrics)

    @functools.cached_property
    def recall(self) -> float:
        """Calculate the overall recall of the predicted tactics across all sessions."""
        return sum(m.recall for m in self.__session_metrics) / len(self.__session_metrics)

    @functools.cached_property
    def f1_score(self) -> float:
        """Calculate the overall F1 score of the predicted tactics across all sessions."""
        return sum(m.f1_score for m in self.__session_metrics) / len(self.__session_metrics)

    @functools.cached_property
    def tactics_precision(self) -> dict[MITRETactic, float]:
        """Calculate precision for each tactic across all sessions."""
        tactic_precisions = {}
        for tactic, (tp, fp, _) in self.__tactics_matches.items():
            tactic_precisions[tactic] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return tactic_precisions

    @functools.cached_property
    def tactics_recall(self) -> dict[MITRETactic, float]:
        """Calculate recall for each tactic across all sessions."""
        tactic_recalls = {}
        for tactic, (tp, _, fn) in self.__tactics_matches.items():
            tactic_recalls[tactic] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return tactic_recalls

    @functools.cached_property
    def tactics_f1_score(self) -> dict[MITRETactic, float]:
        """Calculate F1 score for each tactic across all sessions."""
        tactic_f1_scores = {}
        for tactic in MITRETactic:
            precision = self.tactics_precision[tactic]
            recall = self.tactics_recall[tactic]
            tactic_f1_scores[tactic] = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )
        return tactic_f1_scores

    @functools.cached_property
    def __tactics_matches(self) -> dict[MITRETactic, tuple[float, float, float]]:
        """Calculate precision for each tactic across all sessions."""
        tactic_matches = {}
        for tactic in MITRETactic:
            tp = 0
            fp = 0
            fn = 0
            for m in self.__session_metrics:
                if tactic in m.pred_labels and tactic in m.true_labels:
                    tp += 1
                elif tactic in m.pred_labels and tactic not in m.true_labels:
                    fp += 1
                elif tactic not in m.pred_labels and tactic in m.true_labels:
                    fn += 1

            tactic_matches[tactic] = (tp, fp, fn)
        return tactic_matches

    @functools.cached_property
    def __session_metrics(self) -> list[SessionTacticsMetrics]:
        """Predict tactics for each session using the predictor."""
        predictor = TacticsPredictor(
            llm=self.__llm,
            prompt_predict_tactics=self.__prompt_predict_tactics,
        )
        pred_tactics = {
            session: predictor.predict_tactics(events) for session, events in self.__y_pred_sessions.items()
        }
        return [
            SessionTacticsMetrics(y_pred, self.__y_true_tactics[session]) for session, y_pred in pred_tactics.items()
        ]
