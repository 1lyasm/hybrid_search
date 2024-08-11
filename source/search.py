import argparse
import os
import cassio
import langchain_community
import langchain_huggingface
import termcolor
from langchain_community.vectorstores import Cassandra
from cassio.table.cql import STANDARD_ANALYZER
from langchain_community.document_loaders.csv_loader import (
    CSVLoader,
)
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore


class Searcher:
    def __init__(self):
        pass


def main():
    parser = argparse.ArgumentParser(
        prog="hybrid_searcher",
        description="Performs hybrid search with the given query",
    )
    parser.add_argument(
        "-q", "--query_string", help="Query string", required=True
    )
    arguments = parser.parse_args()

    searcher = Searcher()


if __name__ == "__main__":
    main()
