"""Store module interface for managing event graphs."""

from abc import ABC, abstractmethod

from ontologx.backend.embeddings import Embeddings
from ontologx.store import GraphDocument, StoreConfig


class Store(ABC):
    """Abstract base class for a store module that manages the storage and retrieval of event graphs."""

    def __init__(self, embeddings: Embeddings, config: StoreConfig) -> None:
        """Initialize the store with the given configuration.

        This method should not do any initialization of the store itself, but rather set up the configuration.
        Expensive init operations should be done in the `initialize` method.

        Args:
            embeddings (Embeddings): Embeddings backend to use for the store.
            config (StoreConfig): Configuration for the store module.

        """
        self._embeddings = embeddings
        self._config = config

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
    def add_evaluation_result(self, measure: str, evaluation: str | float) -> None:
        """Add the results of the experiment to the graph store.

        The results are added as properties of the run node.

        Args:
            measure (str): The name of the measure.
            evaluation (Any): The value of the evaluation.

        """

    @abstractmethod
    def add_hyperparameter(self, name: str, value: str | float | bool) -> None:
        """Add a hyperparameter to the schema."""
