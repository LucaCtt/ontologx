"""Neo4j store implementation."""

from langchain_core.embeddings import Embeddings
from neo4j import GraphDatabase

from ontologx.store import GraphDocument, Store, StoreConfig
from ontologx.store.neo4j.dataset import Dataset
from ontologx.store.neo4j.ontology import Ontology
from ontologx.store.neo4j.schema import Schema


class Neo4jStore(Store):
    """A store implementation for Neo4j graph database."""

    def __init__(
        self,
        embeddings: Embeddings,
        config: StoreConfig,
    ) -> None:
        super().__init__(embeddings, config)
        self.__driver = GraphDatabase.driver(config.auth.url, auth=(config.auth.username, config.auth.password))

        self.__onto = Ontology(self.__driver, config.ontology_path)
        self.__schema = Schema(self.__driver, config.study_uri, config.experiment_uri, config.run_uri)
        self.__dataset = Dataset(self.__driver, self._embeddings, config)

    def initialize(self) -> None:
        """Initialize the Neo4j graph database by creating necessary nodes, relationships, constraints, and indexes."""
        self.__onto.initialize()
        self.__schema.initialize()
        self.__dataset.initialize()

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.__driver:
            self.__driver.close()

    def tests(self) -> list[GraphDocument]:
        """Return a list of test documents from the dataset."""
        return self.__dataset.tests()

    def search(
        self,
        criterion: str,
        event: str,
        context: dict | None = None,
        **kwargs: str | float,
    ) -> list[GraphDocument]:
        """Search for events in the dataset based on the given criterion.

        Currently supports 'mmr' for maximum marginal relevance search.

        Args:
            criterion: The search criterion to use.
            event: The event to search for.
            context: Additional context for the search.
            **kwargs: Additional keyword arguments for the search.

        Returns:
            A list of GraphDocument objects representing the search results.

        """
        if criterion == "mmr":
            return self.__dataset.events_mmr_search(event, context, **kwargs)  # type: ignore[call-arg]

        msg = f"Unknown search criterion: {criterion}"
        raise ValueError(msg)

    def add_event_graph(self, event_graph: GraphDocument) -> None:
        """Add an event graph to the dataset."""
        self.__dataset.add_event_graph(event_graph)

    def ontology(self) -> GraphDocument:
        """Return the ontology graph as a GraphDocument."""
        return self.__onto.graph()

    def add_evaluation_result(self, measure: str, evaluation: str | float) -> None:
        """Add an evaluation result to the schema."""
        self.__schema.add_evaluation_result(measure, evaluation)

    def add_hyperparameter(self, name: str, value: str | float | bool) -> None:
        """Add a hyperparameter to the schema."""
        self.__schema.add_hyperparameter(name, value)
