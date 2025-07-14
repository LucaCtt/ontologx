from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph

from ontologx.config import Config
from ontologx.store import GraphDocument, Store
from ontologx.store.neo4j.dataset import Dataset
from ontologx.store.neo4j.ontology import Ontology
from ontologx.store.neo4j.schema import Schema


class Neo4jStore(Store):
    def __init__(self, config: Config, embeddings: Embeddings) -> None:
        self.__embeddings = embeddings
        self.__graph_store = Neo4jGraph(
            url=config.neo4j_url,
            username=config.neo4j_username,
            password=config.neo4j_password,
        )

        self.__onto = Ontology(config, self.__graph_store)
        self.__schema = Schema(config, self.__graph_store)
        self.__dataset = Dataset(config, self.__graph_store, self.__embeddings)

    def initialize(self) -> None:
        self.__onto.initialize()
        self.__schema.initialize()
        self.__dataset.initialize()

    def clear(self) -> None:
        # Delete all nodes and relationships in the graph
        self.__graph_store.query("MATCH (n) DETACH DELETE n")

        # Delete all constraints
        constraints = self.__graph_store.query("SHOW CONSTRAINTS YIELD name RETURN name")
        for constraint in constraints:
            self.__graph_store.query(f"DROP CONSTRAINT {constraint['name']}")

        # Delete all indexes
        indexes = self.__graph_store.query("SHOW INDEXES YIELD name RETURN name")
        for index in indexes:
            self.__graph_store.query(f"DROP INDEX {index['name']}")

    def ontology(self) -> GraphDocument:
        return self.__onto.graph()

    def tests(self) -> list[GraphDocument]:
        return self.__dataset.tests()

    def search(
        self,
        criterion: str,
        event: str,
        context: dict | None = None,
        **kwargs: str | float,
    ) -> list[GraphDocument]:
        if criterion == "mmr":
            return self.__dataset.events_mmr_search(event, context, **kwargs)  # type: ignore[call-arg]

        msg = f"Unknown search criterion: {criterion}"
        raise ValueError(msg)

    def add_event_graph(self, event_graph: GraphDocument) -> None:
        self.__dataset.add_event_graph(event_graph)

    def total_constraints(self) -> int:
        return self.__onto.total_constraints()

    def validate_event_graph(self, event_graph: GraphDocument) -> int:
        return self.__onto.validate(event_graph)

    def add_evaluation_result(self, measure: str, evaluation: str | float) -> None:
        self.__schema.add_evaluation_result(measure, evaluation)
