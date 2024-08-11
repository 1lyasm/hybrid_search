import argparse
import os
import langchain_community
import langchain_huggingface
import termcolor
import json
from langchain.retrievers import (
    EnsembleRetriever,
)
from langchain_community.retrievers import (
    BM25Retriever,
)
from langchain_community.vectorstores import (
    FAISS,
)
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)
from langchain_community.document_loaders.csv_loader import (
    CSVLoader,
)
from langchain_ollama import OllamaLLM


class Searcher:
    def __init__(
        self,
    ):
        documents = CSVLoader(
            os.path.join(
                ".",
                "data",
                "imdb_dataset.csv",
            )
        ).load()

        self.bm25_retriever = BM25Retriever.from_documents(
            documents
        )

        embeddings = HuggingFaceEmbeddings(show_progress=True)
        persist_directory = os.path.join(
            ".",
            "storage",
        )
        if os.path.exists(persist_directory):
            self.faiss_vectorstore = FAISS.load_local(
                persist_directory,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self.faiss_vectorstore = FAISS.from_documents(
                documents,
                embeddings,
            )
            self.faiss_vectorstore.save_local(persist_directory)
        self.faiss_retriever = (
            self.faiss_vectorstore.as_retriever()
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.bm25_retriever,
                self.faiss_retriever,
            ],
            weights=[
                0.5,
                0.5,
            ],
        )

    def search_lexical(
        self,
        query_string,
    ):
        return self.bm25_retriever.invoke(query_string)

    def search_semantic(
        self,
        query_string,
    ):
        return self.faiss_retriever.invoke(query_string)

    def search_hybrid(
        self,
        query_string,
    ):
        return self.ensemble_retriever.invoke(query_string)


def main():
    parser = argparse.ArgumentParser(
        prog="hybrid_searcher",
        description="Performs hybrid search with the given query",
    )
    parser.add_argument(
        "-q",
        "--query_string",
        help="Query string",
        required=True,
    )
    arguments = parser.parse_args()

    output_directory = "output"
    model = OllamaLLM(model="llama3")

    searcher = Searcher()
    function_list = [
        searcher.search_lexical,
        searcher.search_semantic,
        searcher.search_hybrid,
    ]
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    for search_function in function_list:
        with open(
            os.path.join(
                output_directory,
                f"{search_function.__name__}.json",
            ),
            "w",
        ) as output_file:
            output_dictionary = dict()
            output_dictionary["query"] = arguments.query_string
            output_dictionary["search_function"] = (
                search_function.__name__
            )
            retrieved_documents = search_function(
                arguments.query_string
            )
            output_dictionary["retrieved_documents"] = [
                {
                    "id": document.metadata["row"],
                    "text": document.page_content,
                }
                for document in retrieved_documents
            ]
            for document in retrieved_documents:
                prompt = (
                    """For the following query and retrieved document pair,
                    generate an evaluation score between 1 and 10 (inclusive)
                    that shows the relevance of the document to the query.
                    Query: """
                    + output_dictionary["query"]
                    + """
                    Document: """
                    + document.page_content
                    + """
                    Your answers should be in this format (without any extra text):
                        {'score': <your_score> (string), 'reasoning': <your reasoning> (string)}"""
                )
                llm_answer = model.invoke(prompt)
                llm_answer_dictionary = json.load(llm_answer)
                output_dictionary["evaluation"] = (
                    llm_answer_dictionary
                )

            print(
                json.dumps(output_dictionary, indent=4),
                file=output_file,
            )


if __name__ == "__main__":
    main()
