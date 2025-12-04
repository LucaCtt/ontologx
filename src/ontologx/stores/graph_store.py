"""GraphDB store module for managing event graphs."""

from string import Template

import requests
from langchain_community.graphs import OntotextGraphDBGraph
from mitreattack.stix20 import Tactic
from rdflib import Graph

__REPO_TTL = """\
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.
@prefix rep: <http://www.openrdf.org/config/repository#>.
@prefix sail: <http://www.openrdf.org/config/sail#>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

<#ontologx> a rep:Repository;
  rep:repositoryID "ontologx";
  rep:repositoryImpl [
    rep:repositoryType "graphdb:SailRepository";
    <http://www.openrdf.org/config/repository/sail#sailImpl> [
      <http://www.ontotext.com/config/graphdb#disable-sameAs> "false";
      sail:sailType "graphdb:Sail";
    ];
  ];
  rdfs:label "ontologx"^^xsd:string.
"""


class GraphStore:
    """Abstract base class for a store module that manages the storage and retrieval of event graphs."""

    def __init__(self, url: str) -> None:
        # Create repository
        requests.post(
            url + "/rest/repositories",
            files={"config": ("repo.ttl", __REPO_TTL, "text/turtle")},
            timeout=10,
        )

        self.__graph = OntotextGraphDBGraph(url + "/repositories/ontologx")

    def get_graph(
        self,
        event: str,
    ) -> Graph:
        """Get the graph for a given event.

        Args:
            event (str): The event to get the graph for.

        Returns:
            Graph: The graph for the event, or an empty graph if not found.

        """
        g = Graph()
        template = Template("""
        PREFIX mlsx: <https://cyberseclab.unibs.it/mlsx/dict#>

        SELECT ?s ?p ?o WHERE {
            <<?s ?p ?o>> mlsx:eventMessage "$event"^^xsd:string .
        }
        """)

        res = self.__graph.query(template.safe_substitute(event=event))
        for s, p, o in res:
            g.add((s, p, o))

        return g

    def add_graph(self, event: str, graph: Graph, tactic: Tactic) -> None:
        """Add a graph to the store.

        Args:
            event (str): The event associated with the graph.
            graph (Graph): The graph to add.
            tactic (Tactic): The MITRE ATT&CK tactic associated with the event.

        """
        template = Template("""
        <<$s $p $o>> mlsx:eventMessage "$event"^^xsd:string ;
            mlsx:tactic "$tactic"^^xsd:string ;
        """)

        embedded_triples = [
            template.safe_substitute(
                s=s,
                p=p,
                o=o,
                event=event,
                tactic=tactic.name,
            )
            for s, p, o in graph
        ]

        self.__graph.query(
            f"""
            PREFIX mlsx: <https://cyberseclab.unibs.it/mlsx/dict#>

            INSERT DATA { {"\n\n".join(embedded_triples)} }
            """,
        )
