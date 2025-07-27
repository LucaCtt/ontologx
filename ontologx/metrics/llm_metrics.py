import functools

from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams

from ontologx.store import GraphDocument


def _stringify_graph(graph: GraphDocument) -> str:
    """Convert a GraphDocument to a string representation."""
    return f"""{"nodes": {[f"id: {node.id}, type: {node.type}, properties: {node.properties}" for node in graph.nodes]},
        "relationships": {[f"source_id: {rel.source.id}, \
                target_id: {rel.target.id}, \
                type: {rel.type}" for rel in graph.relationships]},
    }
    """


class GEvalGraphAlignmentMetrics:
    @functools.cached_property
    def __measures(self) -> list[float]:
        result = []
        for test_case in self.__test_cases:
            # If the actual output is None, skip the LLM evaluation completely
            if test_case.actual_output == "[]":
                result.append(0)
                continue

            # Re-run the measure for 5 attempts to mitigate invalid LLM output issues
            res = 0
            for _ in range(5):
                try:
                    res = self.__metric.measure(test_case, _show_indicator=False)
                    break
                except ValueError as _:
                    continue

            result.append(res)

        return result

    @functools.cached_property
    def mean(self) -> float:
        """Calculate the mean of the G-Eval measures."""
        return sum(self.__measures) / len(self.__measures) if self.__measures else 0

    @functools.cached_property
    def mean_with_compliance(self) -> float:
        """Calculate the mean of the G-Eval measures, excluding invalid outputs."""
        return (
            sum(compliance * measure for measure, compliance in zip(self.__measures, self.__y_compliance, strict=True))
            / len(self.__measures)
            if self.__measures
            else 0
        )

    def __init__(self, y_pred: list[GraphDocument], y_compliance: list[float], llm_model: DeepEvalBaseLLM):
        """Initialize the G-Eval metrics with predictions and the LLM model.

        Args:
            y_pred (list[GraphDocument]): List of predicted GraphDocuments.
            y_compliance (list[bool]): List of SHACL compliance values for the predicted graphs.
                A value of 1.0 indicates full SHACL compliance, while 0.0 indicates full non-compliance.
            llm_model (DeepEvalBaseLLM): The LLM model used for G-Eval.

        """
        self.__y_compliance = y_compliance
        self.__metric = GEval(
            name="Graph Alignment",
            model=llm_model,
            evaluation_steps=[
                (
                    "Write a detailed description of the input log event in natural language. "
                    "Include what occurred, the involved entities, their roles, any parameters, "
                    "timestamps, or contextual details conveyed in the log."
                ),
                (
                    "Translate the output knowledge graph into a coherent natural language description "
                    "that reflects the event(s) they represent."
                ),
                (
                    "Assess whether the graph description semantically captures "
                    "the same core information as the log event. "
                    "Check for:\n"
                    "  - Coverage: Are all key elements from the log event present?\n"
                    "  - Correctness: Are entities, actions, and relationships represented accurately?\n"
                    "  - Relevance: Are any additional triples relevant to the log event context?\n"
                    "It is acceptable if the graph contains more information than the log event, "
                    "as long as it enriches the representation without introducing unrelated or incorrect content."
                ),
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        )

        self.__test_cases = [
            LLMTestCase(
                input=graph.source.page_content,
                actual_output=_stringify_graph(graph),
            )
            for graph in y_pred
        ]


class TTPMetrics:
    pass
