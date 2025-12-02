"""Module for managing event graphs in a Neo4j store."""

import functools
import uuid
from pathlib import Path

import neo4j
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType

from ontologx.store import Graph, Node, Relationship
from ontologx.store.config import StoreConfig
from ontologx.store.neo4j.utils import get_uri_from_ttl, normalize_input_graph, normalize_output_graph


def _compose_embeddings_text(event: str) -> str:
    # Note: the following characters will be stripped out by langchain-neo4j:
    # + - = && || > < ! ( ) { } [ ] ^ " ~ * ? : \ /
    return f"event is '{event}'"


class Dataset:
    """The Dataset module is responsible for managing the event graphs in the store.

    Includes the loading of the examples and tests, and the search for similar events.
    """

    def __init__(self, graph_store: Neo4jGraph, embeddings: Embeddings, config: StoreConfig) -> None:
        self.__graph_store = graph_store
        self.__embeddings = embeddings
        self.__config = config

        self.__vector_index = Neo4jVector(
            embedding=self.__embeddings,
            username=config.auth.username,
            password=config.auth.password,
            url=config.auth.url,
            index_name="eventsVectorIndex",
            keyword_index_name="eventsKeywordIndex",
            node_label="mlsx__DatasetRow",  # The embedding is not stored on the Event itself to keep it clean
            embedding_node_property="embedding",
            search_type=SearchType.HYBRID,
            retrieval_query="""
            RETURN node.mlsx__eventMessage AS text,
            score,
            {uri: node.uri, _embedding_: node.embedding} AS metadata
            """,
        )

    def initialize(self) -> None:
        """Initialize the dataset by loading the examples and tests, and creating the vector index for events."""
        # Check if the examples are already loaded
        result = self.__graph_store.query(
            """
            MATCH (r:mlsx__Run {uri: $run_uri})-[:mlsx__hasInput]->(d:mlsx__ExampleDataset)
            RETURN d
            LIMIT 1
            """,
            params={"run_uri": self.__config.run_uri},
        )
        if result:
            return

        # Load the examples
        # Note: n10s will not load the examples if they are already present in the store.
        # This is ok, as we want to load the examples again only when they change.
        self.__graph_store.query(
            "CALL n10s.rdf.import.inline($examples, 'Turtle')",
            params={"examples": Path(self.__config.examples_path).read_text()},
        )

        # Attach the examples to the current run
        self.__graph_store.query(
            """
            MATCH (d:mlsx__ExampleDataset), (r:mlsx__Run {uri: $run_uri})
            CREATE (r)-[:mlsx__hasInput]->(d)
            """,
            params={"run_uri": self.__config.run_uri},
        )

        # Create the index for the event messages
        self.__vector_index.create_new_index()
        self.__vector_index.create_new_keyword_index(["mlsx__eventMessage"])

        # Populate the embeddings for the examples
        to_populate = self.__graph_store.query(
            """
            MATCH (d:mlsx__ExampleDataset)-[:mlsx__hasPart]->(r:mlsx__DatasetRow)
            WHERE r.embedding IS NULL
            RETURN elementId(r) AS id, r.mlsx__eventMessage AS eventMessage
            """,
        )

        texts = [_compose_embeddings_text(el["eventMessage"]) for el in to_populate]

        text_embeddings = self.__embeddings.embed_documents(texts)
        self.__graph_store.query(
            """
            UNWIND $data AS row
            MATCH (r:mlsx__DatasetRow)
            WHERE elementId(r) = row.id
            CALL db.create.setNodeVectorProperty(r, 'embedding', row.embedding)
            """,
            params={
                "data": [
                    {"id": el["id"], "embedding": embedding}
                    for el, embedding in zip(to_populate, text_embeddings, strict=True)
                ],
            },
        )

        # Load the tests
        # Note: the test events should not have an embedding.
        self.__graph_store.query(
            "CALL n10s.rdf.import.inline($logs, 'Turtle')",
            params={"logs": Path(self.__config.logs_path).read_text()},
        )
        self.__graph_store.query(
            """
            MATCH (d:mlsx__TestDataset), (r:mlsx__Run {uri: $run_uri})
            MERGE (r)-[:mlsx__hasInput]->(d)
            """,
            params={
                "run_uri": self.__config.run_uri,
            },
        )
        self.__graph_store.query(
            """
            MATCH (d:mlsx__TestDataset)-[:mlsx__hasPart]->(r:mlsx__DatasetRow)-[:mlsx__hasGraph]->(e)
            WHERE r.embedding IS NULL
            SET r.embedding = ''
            """,
        )

    def tests(self) -> list[Graph]:
        """Return a list of test documents from the dataset."""
        test_nodes = self.__graph_store.query(
            """
            MATCH (d:mlsx__TestDataset)-[:mlsx__hasPart]->(r:mlsx__DatasetRow)-[:mlsx__hasGraph]->(e)
            RETURN r.mlsx__eventMessage as eventMessage, e.uri as uri
            ORDER BY e.uri
            """,
        )
        tests = []
        for test in test_nodes:
            graph = self.__get_subgraph_from_node(test["uri"])

            graph.source_event = test["eventMessage"]
            tests.append(normalize_output_graph(graph))

        return tests

    def add_event_graph(self, graph: Graph) -> None:
        """Add an event graph to the store.

        All the nodes will be tagged with the current run name,
        and for Event nodes the embedding will be added.

        Args:
            graph (GraphDocument): The event graph to add.

        """
        # Ensure all nodes have a unique ID and a URI property.
        # Do this before normalizing the graph, so the un-normalized graph
        # has consistent IDs and URIs if it is used somewhere else.
        for node in graph.nodes:
            node.id = f"{self.__config.run_uri}/{uuid.uuid4()}"

        norm_graph = normalize_input_graph(graph)

        # Check if result dataset exists, otherwise create it
        result_dataset = self.__graph_store.query(
            """
            MATCH (r:mlsx__Run {uri: $run_uri})-[:mlsx__hasOutput]->(d:mlsx__OutputDataset)
            RETURN d
            """,
            params={"run_uri": self.__config.run_uri},
        )
        if not result_dataset:
            self.__graph_store.query(
                """
                MATCH (r:mlsx__Run {uri: $run_uri})
                CREATE (d:mlsx__OutputDataset {uri: $out_dataset_uri})<-[:mlsx__hasOutput]-(r)
                """,
                params={"run_uri": self.__config.run_uri, "out_dataset_uri": self.__config.run_uri + "/out-dataset"},
            )

        event_node = next(node for node in norm_graph.nodes if node.type == "olx__Event")
        event_node.id = f"{self.__config.run_uri}/{uuid.uuid4()}"
        event_node.properties["uri"] = event_node.id
        for node in norm_graph.nodes:
            node.id = f"{self.__config.run_uri}/{uuid.uuid4()}"
            node.properties["uri"] = node.id

        text = _compose_embeddings_text(
            norm_graph.source_event,
        )

        dataset_row_properties = {
            "mlsx__eventMessage": graph.source_event,
            "embedding": self.__embeddings.embed_query(text),
            "uri": f"{self.__config.run_uri}/{uuid.uuid4()}",
        }
        self.__graph_store.query(
            """
            MATCH (d:mlsx__OutputDataset)
            WHERE d.uri STARTS WITH $out_dataset_uri
            CREATE (d)-[:mlsx__hasPart]->(r:mlsx__DatasetRow $row_props)
                -[:mlsx__hasGraph]->(n:olx__Event $event_props)
            """,
            params={
                "event_props": event_node.properties,
                "row_props": dataset_row_properties,
                "out_dataset_uri": self.__config.run_uri + "/out-dataset",
            },
        )

        for node in norm_graph.nodes:
            if node.type == "olx__Event":
                continue

            self.__graph_store.query(
                f"CREATE (n:{node.type} $props)",
                params={"props": node.properties},
            )

        for relationship in norm_graph.relationships:
            self.__graph_store.query(
                f"""
                MATCH (a {{uri: $source_uri}}), (b {{uri: $target_uri}})
                CREATE (a)-[:{relationship.type}]->(b)
                """,
                params={
                    "source_uri": relationship.source.id,
                    "target_uri": relationship.target.id,
                },
            )

    def events_mmr_search(
        self,
        event: str,
        k: int = 3,
        fetch_k: int = 30,
        lambda_mult: float = 0.5,
    ) -> list[Graph]:
        """Search for similar events in the store.

        Args:
            event (str): The event message to search for.
            k (int): The number of events to return.
            fetch_k (int): The number of events to pass to the MMR algorithm.
            lambda_mult (float): number between 0 and 1, that determines the trade-off between relevance and diversity.
                0 means maximum diversity, 1 means maximum relevance.

        Returns:
            list[Graph]: The list of graphs of similar events,
                with the nodes they are connected to and their relationships.

        """
        # This filter is for retrieving examples + generated events.
        # Examples will have no run name but an embedding,
        # Tests will have neither. Generated events will have both.
        uri_filter = [self.__examples_uri, self.__config.run_uri]

        query = _compose_embeddings_text(event)
        query_embedding = self.__embeddings.embed_query(query)

        got_docs = self.__vector_index.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            query=query,
            k=fetch_k,
            return_embeddings=True,
        )
        got_docs = [(doc, score) for doc, score in got_docs if doc.metadata["uri"].startswith(tuple(uri_filter))]

        # Get the embeddings for the fetched documents
        got_embeddings = [doc.metadata["_embedding_"] for doc, _ in got_docs]

        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding),
            got_embeddings,
            lambda_mult=lambda_mult,
            k=k,
        )
        selected_docs = [got_docs[i][0] for i in selected_indices]

        # Remove embedding values from metadata
        for doc in selected_docs:
            del doc.metadata["_embedding_"]

        return [normalize_output_graph(self.__get_subgraph_from_node(doc.metadata["uri"])) for doc in selected_docs]

    def __get_subgraph_from_node(self, node_uri: str) -> Graph:
        """Get the subgraph of a node in the store.

        The subgraph will contain all the nodes and relationships connected to the given node, even indirectly.

        Args:
            node_uri (str): The URI of the node to get the subgraph from.

        Returns:
            GraphDocument: The subgraph of the node, with nodes and relationships.

        Raises:
            ValueError: If no subgraph is found for the given node URI.

        """
        # Ugly but quite efficient.
        nodes_subgraphs = self.__graph_store.query(
            """
            MATCH (n {uri: $node_uri})
            CALL apoc.path.subgraphAll(n,
                {labelFilter: '-mlsx__OutputDataset|-mlsx__ExampleDataset|-mlsx__TestDataset'})
            YIELD nodes, relationships
            RETURN
            [node IN nodes | {
            uri: node.uri,
            type: HEAD([label IN LABELS(node) WHERE label <> 'Resource']),
            properties: PROPERTIES(node)
            }] AS nodes,
            [rel IN relationships | {
            source: STARTNODE(rel).uri,
            target: ENDNODE(rel).uri,
            type: TYPE(rel)
            }] AS relationships
            """,
            params={"node_uri": node_uri},
        )

        if not nodes_subgraphs:
            msg = f"No subgraph found for node with URI: {node_uri}"
            raise ValueError(msg)

        nodes_subgraph = nodes_subgraphs[0]

        # Remove the DatasetRow node, as it is not needed in the output graph.
        # However it contains the event message, so it is used as the source of the graph.
        dataset_row_node = next(node for node in nodes_subgraph["nodes"] if node["type"] == "mlsx__DatasetRow")
        nodes_subgraph["nodes"].remove(dataset_row_node)
        nodes_subgraph["relationships"] = [
            relationship
            for relationship in nodes_subgraph["relationships"]
            if relationship["source"] != dataset_row_node["uri"] and relationship["target"] != dataset_row_node["uri"]
        ]

        # The neo4j date and time objects are quite problematic, as they are not JSON serializable.
        # This is a workaround to convert them to strings.
        for node in nodes_subgraph["nodes"]:
            for key, value in node["properties"].items():
                if isinstance(value, neo4j.time.DateTime | neo4j.time.Date | neo4j.time.Time):
                    node["properties"][key] = value.iso_format()

        def get_node_id(uri: str) -> str:
            """Get the node id from the node uri."""
            final_str = uri.split("/")[-1]

            if "#" in final_str:
                return final_str.split("#")[-1]

            return final_str

        nodes_dict = {
            node["uri"]: Node(id=get_node_id(node["uri"]), type=node["type"], properties=node["properties"])
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

        return Graph(
            nodes=list(nodes_dict.values()),
            relationships=relationships,
            source_event=dataset_row_node["properties"]["mlsx__eventMessage"],
        )

    @functools.cached_property
    def __examples_uri(self) -> str:
        """Return the examples URI from the configuration."""
        return get_uri_from_ttl(self.__config.examples_path)
