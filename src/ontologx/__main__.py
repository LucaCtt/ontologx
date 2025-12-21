"""Main entry point for the log knowledge graph builder.

Handles the instantiation of the parser and the backend,
the reading of the logs, and the construction of the knowledge graph.
"""

import logging

import boto3
import polars as pl
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_community.embeddings import InfinityEmbeddings
from rdflib import Graph
from rich.logging import RichHandler

from ontologx.agents.graph_connector import GraphConnectorContext, GraphConnectorInput, graph_connector_agent
from ontologx.settings import Settings
from ontologx.stores import GraphStore, VectorStore

# Setup logging
logging.basicConfig(
    format="%(message)s",
    handlers=[
        RichHandler(
            locals_max_string=200,
            tracebacks_code_width=200,
            tracebacks_width=None,
            omit_repeated_times=False,
            show_path=False,
        ),
    ],
)
logger = logging.getLogger("rich")
logger.setLevel(logging.DEBUG)
# Silence useless logs from Neo4j
logging.getLogger("neo4j").setLevel(logging.ERROR)

settings = Settings()

sts_client = boto3.client(
    "sts",
    config=Config(read_timeout=300),
    aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
    aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
)

# Assume the role
response = sts_client.assume_role(
    RoleArn=settings.aws_role_arn.get_secret_value() if settings.aws_role_arn else None,
    RoleSessionName="langchain-bedrock-session",
    DurationSeconds=60 * 60 * 10,  # 10 hours
)

# Extract the temporary credentials
credentials = response["Credentials"]

llm = ChatBedrockConverse(
    model=settings.llm_name,
    temperature=settings.llm_temperature,
    region_name=settings.aws_region,
    aws_access_key_id=credentials["AccessKeyId"],
    aws_secret_access_key=credentials["SecretAccessKey"],
    aws_session_token=credentials["SessionToken"],
)

embeddings = InfinityEmbeddings(model=settings.embeddings_name, infinity_api_url=settings.embeddings_url)

graph_store = GraphStore(url=settings.graph_database_url)
vector_store = VectorStore(embeddings=embeddings, url=settings.vector_database_url)


def main() -> None:
    """Entry point for the log knowledge graph builder."""
    examples = pl.read_csv(settings.examples_path + "/index.csv")
    examples = examples.rename({col: col.lower() for col in examples.columns})

    # Load example events into the vector store
    vector_store.add_events(
        [
            (row["log_event"], {"device": row["device"], "file_name": row["file_name"]})
            for row in examples.rows(named=True)
        ],
    )

    # Load example graphs into the graph store
    for example in examples.rows(named=True):
        g = Graph()
        g.parse(settings.examples_path + "/" + example["graph_path"])

        graph_store.add_graph(example["log_event"], g)

    # Load the ontology
    ontology = Graph()
    ontology.parse(settings.ontology_path)

    # Read the log events
    events = pl.read_csv(settings.logs_path)
    events = events.rename({col: col.lower() for col in events.columns})

    """Run the log knowledge graph builder."""
    graph_connector_agent.invoke(
        input=GraphConnectorInput(events=events),
        context=GraphConnectorContext(llm=llm, ontology=ontology, vector_store=vector_store, graph_store=graph_store),
        config={"recursion_limit": 500},
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        vector_store.close()
