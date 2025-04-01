from langchain_core.embeddings import Embeddings

from ontologx.config import Config
from ontologx.store.dataset import Dataset
from ontologx.store.driver import Driver
from ontologx.store.module import StoreModule
from ontologx.store.ontology import Ontology


class Store(StoreModule):
    def __init__(self, config: Config, embeddings: Embeddings) -> None:
        super().__init__(config)
        self.__embeddings = embeddings

        self.driver = Driver(self._config)
        self.ontology = Ontology(
            self._config,
            self.driver,
        )
        self.dataset = Dataset(self._config, self.driver, self.__embeddings)

    def initialize(self) -> None:
        self.driver.initialize()
        self.ontology.initialize()
        self.dataset.initialize()

    def clear(self) -> None:
        self.driver.clear()
        self.ontology.clear()
        self.dataset.clear()
