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

        self.schema = Schema(config, self.__graph_store)
        self.ontology = Ontology(config, self.__graph_store)
        self.dataset = Dataset(config, self.__graph_store, self.__embeddings)

    def initialize(self) -> None:
        self.schema.initialize()
        self.ontology.initialize()
        self.dataset.initialize()

    def clear(self) -> None:
        self.schema.clear()
        self.ontology.clear()
        self.dataset.clear()
