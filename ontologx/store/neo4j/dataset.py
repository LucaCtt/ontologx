"""Module for managing event graphs in a Neo4j store."""

import functools
import uuid
from pathlib import Path

import neo4j
import numpy as np
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from neo4j import Driver
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index, upsert_vectors
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.types import RetrieverResultItem

from ontologx.store import GraphDocument, Node, Relationship
from ontologx.store.config import StoreConfig
from ontologx.store.neo4j.utils import get_uri_from_ttl, normalize_input_graph, normalize_output_graph


def _compose_embeddings_text(event: str, context: dict[str, str]) -> str:
    text = f"event: '{event.replace("'", "\\'")}'"

    for key, value in context.items():
        text += f", {key}: '{value.replace("'", "\\'")}'"

    return text


class _LangchainEmbedder(Embedder):
    """Custom embedder to use the LangChain Embeddings interface."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self._embeddings.embed_query(text)


class Dataset:
    """The Dataset module is responsible for managing the event graphs in the store.

    Includes the loading of the examples and tests, and the search for similar events.
    """

    def __init__(self, driver: Driver, embeddings: Embeddings, config: StoreConfig) -> None:
        self.__driver = driver
        self.__embeddings = embeddings
        self.__config = config

        self.__retriever = HybridRetriever(
            driver,
            "eventsVectorIndex",
            "eventsTextIndex",
            _LangchainEmbedder(embeddings),
            result_formatter=lambda x: RetrieverResultItem(
                content=x["uri"],
                metadata={
                    "score": x["score"],
                    "embedding": x["embedding"],
                },
            ),
        )

    def initialize(self) -> None:
        """Initialize the dataset by loading the examples and tests, and creating the vector index for events."""
        # Check if the examples are already loaded
        dataset, _, _ = self.__driver.execute_query(
            """
            MATCH (r:mlsx__Run {uri: $run_uri})-[:mlsx__hasInput]->(d:mlsx__ExampleDataset)
            RETURN d
            LIMIT 1
            """,
            run_uri=self.__config.run_uri,
        )
        if dataset:
            return

        # Load the examples
        # Note: n10s will not load the examples if they are already present in the store.
        # This is ok, as we want to load the examples again only when they change.
        self.__driver.execute_query(
            "CALL n10s.rdf.import.inline($examples, 'Turtle')",
            examples=Path(self.__config.examples_path).read_text(),
        )

        # Attach the examples to the current run
        self.__driver.execute_query(
            """
            MATCH (d:mlsx__ExampleDataset), (r:mlsx__Run {uri: $run_uri})
            CREATE (r)-[:mlsx__hasInput]->(d)
            """,
            run_uri=self.__config.run_uri,
        )

        # Get examples to populate embeddings
        to_populate, _, _ = self.__driver.execute_query(
            """
            MATCH (d:mlsx__ExampleDataset)-[:mlsx__hasPart]->(r:mlsx__DatasetRow)
            WHERE r.embedding IS NULL
            OPTIONAL MATCH (r)-[:mlsx__hasContext]->(s:olx__Source)
            RETURN elementId(r) AS id, r.mlsx__eventMessage AS eventMessage, s.olx__sourceName AS sourceName,
            s.olx__sourceDevice AS sourceDevice
            """,
        )

        texts = []
        for el in to_populate:
            context = {}
            if el.get("sourceName"):
                context["sourceName"] = el["sourceName"]
            if el.get("sourceDevice"):
                context["sourceDevice"] = el["sourceDevice"]

            texts.append(_compose_embeddings_text(el["eventMessage"], context))

        text_embeddings = self.__embeddings.embed_documents(texts)

        # Get the dimensions of the embeddings
        embeddings_dimensions = len(text_embeddings[0])

        # Create the index for the event messages
        create_vector_index(
            driver=self.__driver,
            name="eventsVectorIndex",
            label="mlsx__DatasetRow",  # The embedding is not stored on the Event itself to keep it clean
            embedding_property="embedding",
            dimensions=embeddings_dimensions,
            similarity_fn="cosine",
        )

        upsert_vectors(
            driver=self.__driver,
            ids=[el["id"] for el in to_populate],
            embedding_property="embedding",
            embeddings=text_embeddings,
        )

        # Create the fulltext index for the event messages
        create_fulltext_index(
            driver=self.__driver,
            name="eventsTextIndex",
            label="mlsx__DatasetRow",
            node_properties=["mlsx__eventMessage"],
        )

        # Load the tests
        # Note: the test events should not have an embedding.
        self.__driver.execute_query(
            "CALL n10s.rdf.import.inline($tests, 'Turtle')",
            tests=Path(self.__config.tests_path).read_text(),
        )
        self.__driver.execute_query(
            """
            MATCH (d:mlsx__TestDataset), (r:mlsx__Run {uri: $run_uri})
            MERGE (r)-[:mlsx__hasInput]->(d)
            """,
            run_uri=self.__config.run_uri,
        )
        self.__driver.execute_query(
            """
            MATCH (d:mlsx__TestDataset)-[:mlsx__hasPart]->(r:mlsx__DatasetRow)-[:mlsx__hasLabel]->(e:olx__Event)
            WHERE r.embedding IS NULL
            SET r.embedding = ''
            """,
        )

    def tests(self) -> list[GraphDocument]:
        """Return a list of test documents from the dataset."""
        test_nodes, _, _ = self.__driver.execute_query(
            """
            MATCH (d:mlsx__TestDataset)-[:mlsx__hasPart]->(r:mlsx__DatasetRow)-[:mlsx__hasLabel]->(e:olx__Event)
            RETURN r.mlsx__eventMessage as eventMessage, e.uri as uri
            ORDER BY e.uri
            """,
        )
        tests = []
        for test in test_nodes:
            graph = self.__get_subgraph_from_node(test["uri"])

            source_node = next((node for node in graph.nodes if node.type == "olx__Source"), None)
            context = {}
            if source_node:
                if source_node.properties.get("olx__sourceName"):
                    context["sourceName"] = source_node.properties["olx__sourceName"]
                if source_node.properties.get("olx__sourceDevice"):
                    context["sourceDevice"] = source_node.properties["olx__sourceDevice"]

            graph.source = Document(page_content=test["eventMessage"], metadata=context)
            tests.append(normalize_output_graph(graph))

        return tests

    def add_event_graph(self, graph: GraphDocument) -> None:
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
        result_dataset, _, _ = self.__driver.execute_query(
            """
            MATCH (r:mlsx__Run {uri: $run_uri})-[:mlsx__hasOutput]->(d:mlsx__OutputDataset)
            RETURN d
            """,
            run_uri=self.__config.run_uri,
        )
        if not result_dataset:
            self.__driver.execute_query(
                """
                MATCH (r:mlsx__Run {uri: $run_uri})
                CREATE (d:mlsx__OutputDataset {uri: $out_dataset_uri})<-[:mlsx__hasOutput]-(r)
                """,
                run_uri=self.__config.run_uri,
                out_dataset_uri=self.__config.run_uri + "/out-dataset",
            )

        event_node = next(node for node in norm_graph.nodes if node.type == "olx__Event")
        event_node.id = f"{self.__config.run_uri}/{uuid.uuid4()}"
        event_node.properties["uri"] = event_node.id
        for node in norm_graph.nodes:
            node.id = f"{self.__config.run_uri}/{uuid.uuid4()}"
            node.properties["uri"] = node.id

        text = _compose_embeddings_text(
            norm_graph.source.page_content,
            norm_graph.source.metadata,
        )

        dataset_row_properties = {
            "eventMessage": graph.source.page_content,
            "embedding": self.__embeddings.embed_query(text),
            "uri": f"{self.__config.run_uri}/{uuid.uuid4()}",
        }
        self.__driver.execute_query(
            """
            MATCH (d:mlsx__OutputDataset)
            WHERE d.uri STARTS WITH $out_dataset_uri
            CREATE (d)-[:mlsx__hasPart]->(r:mlsx__DatasetRow $row_props)
                -[:mlsx__hasLabel]->(n:olx__Event $event_props)
            """,
            event_props=event_node.properties,
            row_props=dataset_row_properties,
            out_dataset_uri=self.__config.run_uri + "/out-dataset",
        )

        for node in norm_graph.nodes:
            if node.type == "olx__Event":
                continue

            self.__driver.execute_query(
                "CALL apoc.create.node($label, $props)",
                label=node.type,
                props=node.properties,
            )

        for relationship in norm_graph.relationships:
            self.__driver.execute_query(
                """
                MATCH (a {uri: $source_uri}), (b {uri: $target_uri})
                CALL apoc.create.relationship(a, $relationship_type, {}, b)
                """,
                source_uri=relationship.source.id,
                relationship_type=relationship.type,
                target_uri=relationship.target.id,
            )

    def events_mmr_search(
        self,
        event: str,
        context: dict | None = None,
        k: int = 3,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[GraphDocument]:
        """Search for similar events in the store.

        Args:
            event (str): The event message to search for.
            context (dict): The context to use for the search.
            k (int): The number of events to return.
            fetch_k (int): The number of events to pass to the MMR algorithm.
            lambda_mult (float): number between 0 and 1, that determines the trade-off between relevance and diversity.
                0 means maximum diversity, 1 means maximum relevance.

        Returns:
            list[GraphDocument]: The list of graphs of similar events,
                with the nodes they are connected to and their relationships.

        """
        query = _compose_embeddings_text(event, context or {})
        query_embedding = self.__embeddings.embed_query(query)
        search_results = self.__retriever.search(
            query=query_embedding,
            top_k=k * 5,  # Fetch more documents to ensure we have enough after filtering
        )
        relevant_docs = search_results.items
        relevant_docs.sort(key=lambda x: x.metadata["score"] if x.metadata else 0, reverse=True)
        relevant_docs = relevant_docs[:fetch_k]  # Limit to fetch_k results

        # Get examples only or examples + generated events,
        # depending on the configuration.
        allowed_uris = [self.__examples_uri]
        if self.__config.generated_graphs_retrieval:
            allowed_uris.append(self.__config.run_uri)

        selected_indexes = maximal_marginal_relevance(
            np.array(query_embedding),
            [
                doc.metadata["embedding"]
                for doc in relevant_docs
                if doc.metadata is not None and "embedding" in doc.metadata
            ],
            k=k,
            lambda_mult=lambda_mult,
        )
        selected_uris = [relevant_docs[i].content for i in selected_indexes]

        return [normalize_output_graph(self.__get_subgraph_from_node(uri)) for uri in selected_uris]

    def __get_subgraph_from_node(self, node_uri: str) -> GraphDocument:
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
        nodes_subgraphs, _, _ = self.__driver.execute_query(
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
            node_uri=node_uri,
        )

        if not nodes_subgraphs:
            msg = f"No subgraph found for node with URI: {node_uri}"
            raise ValueError(msg)

        nodes_subgraph = dict(nodes_subgraphs[0])

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

        # Create the context from the event source, if present.
        source_node = next((node for node in nodes_dict.values() if node.type == "olx__Source"), None)
        context = {key: value for key, value in source_node.properties.items() if key != "uri"} if source_node else {}

        return GraphDocument(
            nodes=list(nodes_dict.values()),
            relationships=relationships,
            source=Document(page_content=dataset_row_node["properties"]["mlsx__eventMessage"], metadata=context),
        )

    @functools.cached_property
    def __examples_uri(self) -> str:
        """Return the examples URI from the configuration."""
        return get_uri_from_ttl(self.__config.examples_path)
