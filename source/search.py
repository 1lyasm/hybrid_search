import argparse
import os
import cassio
import langchain_community
import langchain_huggingface
import termcolor
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import (
    CSVLoader,
)


class Searcher:
    def __init__(self):
        documents = CSVLoader(
            os.path.join(".", "data", "imdb_dataset.csv")
        ).load()

        self.bm25_retriever = BM25Retriever.from_documents(
            documents
        )

        embeddings = HuggingFaceEmbeddings(show_progress=True)
        persist_directory = os.path.join(".", "storage")
        if os.path.exists(persist_directory):
            self.faiss_vectorstore = FAISS.load_local(
                persist_directory,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.faiss_vectorstore = FAISS.from_documents(
                documents, embeddings
            )
            self.faiss_vectorstore.save_local(persist_directory)
        self.faiss_retriever = self.faiss_vectorstore.as_retriever()

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.bm25_retriever,
                self.faiss_retriever,
            ],
            weights=[0.5, 0.5],
        )

    def search_lexical(self, query_string):
        return self.bm25_retriever.invoke(query_string)

    def search_semantic(self, query_string):
        return self.faiss_retriever.invoke(query_string)

    def search_hybrid(self, query_string):
        return self.ensemble_retriever.invoke(query_string)


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
        f"Lexical search result: {searcher.search_lexical(arguments.query_string)}",
        "blue",
    )
    termcolor.cprint(
        f"Semantic search result: {searcher.search_semantic(arguments.query_string)}",
        "blue",
    )
    termcolor.cprint(
        f"Hybrid search result: {searcher.search_hybrid(arguments.query_string)}",
        "blue",
    )


if __name__ == "__main__":
    main()
