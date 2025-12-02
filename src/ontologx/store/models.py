"""Models for representing nodes, relationships, and graphs in the OntologX store."""

from dataclasses import dataclass
from typing import Any

from mitreattack.stix20 import Tactic


@dataclass
class Node:
    """A node in the graph representing an entity."""

    id: str
    type: str
    properties: dict[str, Any]

    tactics: list[Tactic] | None = None


@dataclass
class Relationship:
    """A relationship between two nodes in the graph."""

    source: Node
    target: Node
    type: str

    tactics: list[Tactic] | None = None


@dataclass
class Graph:
    """A graph consisting of nodes and relationships."""

    nodes: list[Node]
    relationships: list[Relationship]

    source_event: str

    tactics: list[Tactic] | None = None
