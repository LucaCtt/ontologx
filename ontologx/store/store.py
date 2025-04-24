from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph

from ontologx.config import Config
from ontologx.store.dataset import Dataset
from ontologx.store.module import StoreModule
from ontologx.store.ontology import Ontology
from ontologx.store.schema import Schema


class Store(StoreModule):
    def __init__(self, config: Config, embeddings: Embeddings) -> None:
        self.__embeddings = embeddings
        self.__graph_store = Neo4jGraph(
            url=config.neo4j_url,
            username=config.neo4j_username,
            password=config.neo4j_password,
        )

        self.ontology = Ontology(config, self.__graph_store)
        self.schema = Schema(config, self.__graph_store)
        self.dataset = Dataset(config, self.__graph_store, self.__embeddings)

    def initialize(self) -> None:
        self.ontology.initialize()
        self.schema.initialize()
        self.dataset.initialize()

    def clear(self) -> None:
        # Delete all nodes and relationships in the graph
        self.__graph_store.query("MATCH (n) DETACH DELETE n")

        # Delete all indexes
        indexes = self.__graph_store.query("SHOW INDEXES YIELD name RETURN name")
        for index in indexes:
            self.__graph_store.query(f"DROP INDEX {index['name']}")

        # Delete all constraints
        constraints = self.__graph_store.query("SHOW CONSTRAINTS YIELD name RETURN name")
        for constraint in constraints:
            self.__graph_store.query(f"DROP CONSTRAINT {constraint['name']}")

        # Delete all triggers
        triggers = self.__graph_store.query("CALL apoc.trigger.list() YIELD name RETURN name")
        for trigger in triggers:
            self.__graph_store.query(f"CALL apoc.trigger.remove('{trigger['name']}')")
