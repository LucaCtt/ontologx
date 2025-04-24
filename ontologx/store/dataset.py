from pathlib import Path
from typing import Any

import neo4j
from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

from ontologx.config import Config
from ontologx.store.module import StoreModule


class Dataset(StoreModule):
    """The Dataset module is responsible for managing the event graphs in the store.

    Includes the loading of the examples and tests, and the search for similar events.
    """

    def __init__(self, config: Config, graph_store: Neo4jGraph, embeddings: Embeddings) -> None:
        self.__config = config
        self.__graph_store = graph_store
        self.__embeddings = embeddings

        self.__vector_index = Neo4jVector(
            embedding=self.__embeddings,
            username=self.__config.neo4j_username,
            password=self.__config.neo4j_password,
            url=self.__config.neo4j_url,
            index_name=self.__config.events_index_name,
            node_label="Event",
            embedding_node_property="embedding",
            retrieval_query="""
            RETURN node.eventMessage AS text,
            score,
            {uri: node.uri, runName: node.runName, _embedding_: node.embedding} AS metadata
            """,
        )

    def initialize(self) -> None:
        # Check if the examples are already loaded
        result = self.__graph_store.query(
            """
            MATCH (r:Run {runName: $run_name})-[:hasInput]->(d:Dataset)
            RETURN d
            LIMIT 1
            """,
            params={"run_name": self.__config.run_name},
        )
        if result:
            return

        # Load the examples and attach them to the current run
        self.__graph_store.query(
            "CALL n10s.rdf.import.inline($examples, 'Turtle')",
            params={"examples": Path(self.__config.examples_path).read_text()},
        )
        self.__graph_store.query(
            """
            MATCH (d:Dataset {type: 'examples'}), (r:Run {runName: $run_name})
            WHERE d.runName IS NULL
            SET d.runName = $run_name
            CREATE (r)-[:hasInput]->(d)
            """,
            params={"run_name": self.__config.run_name},
        )

        # Create the index for the event messages
        self.__vector_index.create_new_index()

        # Populate the embeddings for the examples
        to_populate = self.__graph_store.query(
            """
            MATCH (d:Dataset {type: 'examples', runName: $run_name})-[:hasPart]->(e:Event)
            WHERE e.embedding IS NULL
            SET e.runName = $run_name
            RETURN elementId(e) AS id, e.eventMessage as eventMessage
            """,
            params={
                "run_name": self.__config.run_name,
                "examples_uri": self.__config.examples_uri,
            },
        )
        text_embeddings = self.__embeddings.embed_documents([el["eventMessage"] for el in to_populate])
        self.__graph_store.query(
            """
            UNWIND $data AS row
            MATCH (e:Event)
            WHERE elementId(e) = row.id
            CALL db.create.setNodeVectorProperty(e, 'embedding', row.embedding)
            """,
            params={
                "data": [
                    {"id": el["id"], "embedding": embedding}
                    for el, embedding in zip(to_populate, text_embeddings, strict=True)
                ],
            },
        )

        # Load the tests
        # Note: the test events should not have an embedding
        # so they are not populated in the index.
        self.__graph_store.query(
            "CALL n10s.rdf.import.inline($tests, 'Turtle')",
            params={"tests": Path(self.__config.tests_path).read_text()},
        )
        self.__graph_store.query(
            """
            MATCH (d: Dataset {type: 'tests'}), (r:Run {runName: $run_name})
            WHERE d.runName IS NULL
            SET d.runName = $run_name
            MERGE (r)-[:hasInput]->(d)
            """,
            params={
                "tests_uri": self.__config.tests_uri,
                "run_name": self.__config.run_name,
            },
        )
        self.__graph_store.query(
            """
            MATCH (d:Dataset {type: 'tests', runName: $run_name})-[:hasPart]->(e:Event)
            SET e.runName = $run_name
            SET e.embedding = NULL
            """,
            params={
                "run_name": self.__config.run_name,
                "tests_uri": self.__config.tests_uri,
            },
        )

    def tests(self) -> list[tuple[str, dict, GraphDocument]]:
        test_nodes = self.__graph_store.query(
            """
            MATCH (d:Dataset {type: 'tests', runName: $run_name})-[:hasPart]->(e:Event)
            ORDER BY e.uri
            RETURN e.eventMessage as message, e.uri as uri
            """,
            params={"run_name": self.__config.run_name},
        )
        tests = []
        for test in test_nodes:
            ground_truth = self.__get_subgraph_from_node(test["uri"])

            source_node = next((node for node in ground_truth.nodes if node.type == "Source"), None)
            context = (
                {"source": source_node.properties["sourceName"], "device": source_node.properties["sourceDevice"]}
                if source_node
                else {}
            )
            tests.append((test["message"], context, ground_truth))

        return tests

    def add_event_graph(self, graph: GraphDocument) -> None:
        """Add an event graph to the store.

        All the nodes will be tagged with the current run name,
        and for Event nodes the embedding will be added.

        Args:
            graph (GraphDocument): The event graph to add.

        """
        for node in graph.nodes:
            # Add the run_name and (for the Event nodes) the embedding.
            additional_properties: dict[str, Any] = {"runName": self.__config.run_name}
            if node.type == "Event":
                # This will raise an exception if the LLM produces an Event node without a message property.
                additional_properties["embedding"] = self.__embeddings.embed_query(node.properties["eventMessage"])

            self.__graph_store.query(
                "CALL apoc.create.node([$type], $props) YIELD node",
                params={"type": node.type, "props": {**node.properties, **additional_properties}},
            )

        for relationship in graph.relationships:
            self.__graph_store.query(
                """
                MATCH (a {uri: $source_uri}), (b {uri: $target_uri})
                CALL apoc.create.relationship(a, $type, {}, b) YIELD rel
                RETURN rel
                """,
                params={
                    "source_uri": relationship.source.id,
                    "target_uri": relationship.target.id,
                    "type": relationship.type,
                },
            )

    def events_mmr_search(
        self,
        event: str,
        k: int = 3,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[tuple[str, GraphDocument]]:
        """Search for similar events in the store.

        Args:
            event (str): The event message to search for.
            k (int): The number of events to return.
            fetch_k (int): The number of events to pass to the MMR algorithm.
            lambda_mult (float): number between 0 and 1, that determines the trade-off between relevance and diversity.
                0 means maximum diversity, 1 means maximum relevance.

        Returns:
            list[GraphDocument]: The list of graphs of similar events,
                with the nodes they are connected to and their relationships.

        """
        relevant_docs = self.__vector_index.max_marginal_relevance_search(
            query=event,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter={"runName": {"$eq": self.__config.run_name}, "embedding": {"$ne": None}},
        )

        return [
            (
                doc.page_content,
                self.__get_subgraph_from_node(doc.metadata["uri"]),
            )
            for doc in relevant_docs
        ]

    def __get_subgraph_from_node(self, node_uri: str, props_to_remove: list[str] | None = None) -> GraphDocument:
        """Get the subgraph of a node in the store.

        The subgraph will contain all the nodes and relationships connected to the given node, even indirectly.
        """
        if props_to_remove is None:
            props_to_remove = []

        props_to_remove = [*props_to_remove, "embedding"]

        # Ugly but quite efficient. Also filters out the embedding property and the Resource label.
        nodes_subgraphs = self.__graph_store.query(
            """
            MATCH (n {uri: $node_uri})
            CALL apoc.path.subgraphAll(n, {labelFilter: '-Dataset'})
            YIELD nodes, relationships
            RETURN
            [node IN nodes | {
            uri: node.uri,
            type: HEAD([label IN LABELS(node) WHERE label <> 'Resource']),
            properties: apoc.map.removeKeys(PROPERTIES(node), $props_to_remove)
            }] AS nodes,
            [rel IN relationships | {
            source: STARTNODE(rel).uri,
            target: ENDNODE(rel).uri,
            type: TYPE(rel)
            }] AS relationships
            """,
            params={"node_uri": node_uri, "props_to_remove": props_to_remove},
        )

        if not nodes_subgraphs:
            return GraphDocument(nodes=[], relationships=[])

        nodes_subgraph = nodes_subgraphs[0]

        # The neo4j date and time objects are quite problematic, as they are not JSON serializable.
        # This is a workaround to convert them to strings.
        for node in nodes_subgraph["nodes"]:
            for key, value in node["properties"].items():
                if isinstance(value, neo4j.time.DateTime):
                    node["properties"][key] = value.iso_format()
                if isinstance(value, neo4j.time.Date):
                    node["properties"][key] = value.iso_format()

        nodes_dict = {
            node["uri"]: Node(id=node["uri"], type=node["type"], properties=node["properties"])
            for node in nodes_subgraph["nodes"]
        }

        relationships = (
            [
                Relationship(
                    source=nodes_dict[relationship["source"]],
                    target=nodes_dict[relationship["target"]],
                    type=relationship["type"],
                )
                for relationship in nodes_subgraph["relationships"]
            ]
            if "relationships" in nodes_subgraph
            else []  # The node may not have any relationships
        )

        return GraphDocument(
            nodes=list(nodes_dict.values()),
            relationships=relationships,
        )
