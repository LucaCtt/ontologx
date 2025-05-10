from abc import ABC, abstractmethod

from ontologx.store import GraphDocument


class StoreModule(ABC):
    """Abstract class for store modules."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the store module by creating module-specific nodes."""


class Store(StoreModule, ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the store module by creating module-specific nodes."""

    @abstractmethod
    def clear(self) -> None:
        """Clear the store back to a clean state."""

    @abstractmethod
    def ontology(self) -> GraphDocument:
        """Get the log ontology."""

    @abstractmethod
    def tests(self) -> list[GraphDocument]:
        """Get the tests set."""

    @abstractmethod
    def add_event_graph(self, event_graph: GraphDocument) -> None:
        """Add an event graph to the store.

        Args:
            event_graph (GraphDocument): The event graph to add.

        """

    @abstractmethod
    def search(self, criterion: str, event: str, **kwargs: str | float) -> list[GraphDocument]:
        """Search for event graphs in the store based on a criterion and event.

        Args:
            criterion (str): The search criterion.
            event (str): The event to search for.
            **kwargs: Additional keyword arguments for the search.

        """

    @abstractmethod
    def add_evaluation_result(self, measure: str, evaluation: str | float) -> None:
        """Add the results of the experiment to the graph store.

        The results are added as properties of the run node.

        Args:
            measure (str): The name of the measure.
            evaluation (Any): The value of the evaluation.

        """
