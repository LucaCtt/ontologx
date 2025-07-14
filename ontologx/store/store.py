from abc import ABC, abstractmethod

from ontologx.store import GraphDocument


class Store(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the store module by creating the necessary nodes, relationships, constraints, and indexes.

        In particular, this method should create the ML Schema, the log ontology, and the datasets.
        This method should be called before any other operations on the store.
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear the store back to a clean state, removing nodes, relationships, constraints, and indexes."""

    @abstractmethod
    def ontology(self) -> GraphDocument:
        """Get the log ontology."""

    @abstractmethod
    def tests(self) -> list[GraphDocument]:
        """Get the test set."""

    @abstractmethod
    def search(
        self,
        criterion: str,
        event: str,
        context: dict | None = None,
        **kwargs: str | float,
    ) -> list[GraphDocument]:
        """Search for event graphs in the store based on a criterion and event.

        Args:
            criterion (str): The search criterion.
            event (str): The event to search for.
            context (dict | None): Optional context for the log event.
            **kwargs: Additional keyword arguments for the search.

        Returns:
            list[GraphDocument]: A list of event graphs that match the search criterion.

        """

    @abstractmethod
    def add_event_graph(self, event_graph: GraphDocument) -> None:
        """Add an event graph to the store.

        Args:
            event_graph (GraphDocument): The event graph to add.

        """

    @abstractmethod
    def total_constraints(self) -> int:
        """Get the total number of SHACL constraints in the store.

        Returns:
            int: The total number of constraints.

        """

    @abstractmethod
    def validate_event_graph(self, event_graph: GraphDocument) -> int:
        """Validate the event graph against the SHACL constraints defined in the ontology.

        Args:
            event_graph (GraphDocument): The event graph to validate with SHACL.

        Returns:
            int: The number of SHACL violations found in the event graph.

        """

    @abstractmethod
    def add_evaluation_result(self, measure: str, evaluation: str | float) -> None:
        """Add the results of the experiment to the graph store.

        The results are added as properties of the run node.

        Args:
            measure (str): The name of the measure.
            evaluation (Any): The value of the evaluation.

        """
