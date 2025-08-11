"""Main entry point for the log knowledge graph builder.

Handles the instantiation of the parser and the backend,
the reading of the logs, and the construction of the knowledge graph.
"""

import logging

from rich.logging import RichHandler

from ontologx.config import Config
from ontologx.run_handler import RunHandler

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

config = Config()


def main() -> None:
    """Run the log knowledge graph builder."""
    logger.info("Experiment: '%s'", config.experiment_name)
    logger.info("Embeddings model: '%s'", config.embeddings_model)
    logger.info("Language model: '%s'", config.parser_model)
    logger.info("Parser type: '%s'", config.parser_type)

    run_handler = RunHandler(config)

    for run_index in range(config.n_runs):
        logger.info("----------------------")
        logger.info("Run %d", run_index + 1)
        run_handler.start_new_run()


if __name__ == "__main__":
    main()
