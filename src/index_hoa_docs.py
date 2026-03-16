import argparse
import logging
from pathlib import Path

import yaml

from data_manipulation import (
    get_data_list,
    clean_document,
    create_text_splitter,
    create_vector_store,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index HOA/strata documents into a FAISS vector store."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML config file (e.g. conf/local/hoa_index.yml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    docs_path = cfg["docs_path"]
    store_name = cfg["vector_store"]["store_name"]
    index_name = cfg["vector_store"]["index_name"]

    logger.info("Indexing HOA documents from %s", docs_path)
    df = get_data_list(docs_path)

    logger.info("Cleaning %d documents", len(df))
    df["Content"] = df["path"].apply(clean_document)

    logger.info("Creating text chunks")
    df = create_text_splitter(df)

    logger.info("Creating and persisting vector store")
    create_vector_store(df, store_name=store_name, index_name=index_name)

    logger.info("Indexing complete")


if __name__ == "__main__":
    main()

