"""Store configuration module for OntologX."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StoreAuth:
    """Configuration for store authentication."""

    url: str
    username: str
    password: str


@dataclass(frozen=True)
class StoreConfig:
    """Configuration for the graph store."""

    study_uri: str

    experiment_uri: str

    run_uri: str

    ontology_path: str

    examples_path: str

    tests_path: str

    generated_graphs_retrieval: bool

    auth: StoreAuth
