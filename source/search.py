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
        persist_directory = os.path.join(".", "storage")
        embeddings = langchain_huggingface.HuggingFaceEmbeddings(
            show_progress=True
        )
        if os.path.exists(persist_directory):
            self.vectorstore = FAISS.load_local(
                persist_directory,
                embeddings,
            )
        else:
            documents = CSVLoader(
                os.path.join(".", "data", "imdb_dataset.csv")
            ).load()
            self.vectorstore = FAISS.from_documents(
                documents,
                embeddings,
            )
            self.vectorstore.save_local(persist_directory)

    def semantic_search(self, query_string):
        return self.vectorstore.as_retriever(
            search_kwargs={"body_search": None}
        ).invoke(query_string)

    def keyword_search(self, query_string):
        return self.vectorstore.as_retriever(
            search_kwargs={"body_search": query_string}
        ).invoke("")

    def hybrid_search(self, query_string):
        return self.vectorstore.as_retriever(
            search_kwargs={"body_search": query_string}
        ).invoke(query_string)


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
    termcolor.cprint(
        f"Keyword-based search result: {searcher.keyword_search(arguments.query_string)}",
        "blue",
    )
    termcolor.cprint(
        f"Semantic search result: {searcher.semantic_search(arguments.query_string)}",
        "blue",
    )
    termcolor.cprint(
        f"Hybrid search result: {searcher.hybrid_search(arguments.query_string)}",
        "blue",
    )


if __name__ == "__main__":
    main()
