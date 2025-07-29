"""SHACL metrics computation module."""

import functools

import pyshacl
import rdflib
from owlrl import DeductiveClosure, RDFS_Semantics

from ontologx.store import GraphDocument


def _expand_namespace_prefix(name: str, namespaces: dict[str, rdflib.URIRef]) -> rdflib.URIRef:
    """Expand a namespace prefix in a node type to a full URI."""
    if ":" not in name:
        return rdflib.URIRef(name)

    prefix, local_name = name.split(":", 1)
    if prefix not in namespaces:
        msg = f"Namespace prefix '{prefix}' not found in the provided namespaces."
        raise ValueError(msg)

    return rdflib.URIRef(namespaces[prefix].toPython() + local_name)


def _convert_to_rdflib(data_graph: GraphDocument, ontology_graph: rdflib.Graph) -> rdflib.Graph:
    """Convert a GraphDocument to an rdflib Graph."""
    res = rdflib.Graph()
    namespaces = dict(ontology_graph.namespace_manager.namespaces())
    namespaces["olx"] = namespaces.pop("")

    for node in data_graph.nodes:
        node_uri = rdflib.URIRef(str(node.id))
        res.add((node_uri, rdflib.RDF.type, _expand_namespace_prefix(node.type, namespaces)))

        for prop, value in node.properties.items():
            if prop == "uri":
                continue

            res.add((node_uri, _expand_namespace_prefix(prop, namespaces), rdflib.Literal(value)))

    for rel in data_graph.relationships:
        start_uri = rdflib.URIRef(str(rel.source.id))
        end_uri = rdflib.URIRef(str(rel.target.id))
        res.add((start_uri, _expand_namespace_prefix(rel.type, namespaces), end_uri))

    DeductiveClosure(RDFS_Semantics).expand(res)

    return res


class SHACLMetrics:
    """Class to calculate SHACL metrics for a given graph."""

    def __init__(self, graph: GraphDocument, ontology_path: str, shacl_path: str) -> None:
        self.__ontology_graph = rdflib.Graph()
        self.__ontology_graph.parse(ontology_path, format="turtle")

        self.__shacl_graph = rdflib.Graph()
        self.__shacl_graph.parse(shacl_path, format="turtle")

        self.__data_graph = _convert_to_rdflib(graph, self.__ontology_graph)

    @functools.cached_property
    def violations_ratio(self) -> float:
        """Calculate the ratio of SHACL violations to total constraints."""
        if self.__total_constraints == 0:
            return 0.0

        return self.__total_violations / self.__total_constraints

    @functools.cached_property
    def compliance_ratio(self) -> float:
        """Calculate the compliance ratio based on SHACL violations."""
        return 1.0 - self.violations_ratio

    @functools.cached_property
    def __total_violations(self) -> int:
        """Validate the graph against SHACL constraints.

        Returns:
            float: number of SHACL constraints that are violated

        """
        _, res_graph, _ = pyshacl.validate(
            self.__data_graph,
            shacl_graph=self.__shacl_graph,
            ont_graph=self.__ontology_graph,
            inference="none",
        )

        if not isinstance(res_graph, rdflib.Graph):
            msg = "Validation results are not in rdflib.Graph format."
            raise TypeError(msg)

        violations = list(res_graph.subject_predicates(rdflib.URIRef("http://www.w3.org/ns/shacl#ValidationResult")))
        return len(violations)

    @functools.cached_property
    def __total_constraints(self) -> int:
        """Get the total number of SHACL constraints."""
        constraints = list(self.__shacl_graph.subject_objects(rdflib.URIRef("http://www.w3.org/ns/shacl#property")))
        return len(constraints)
