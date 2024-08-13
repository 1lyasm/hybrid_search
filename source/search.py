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
    def __init__(self, arguments):
        retrieved_document_count = 10

        documents = CSVLoader(arguments.dataset_path).load()

        self.bm25_retriever = BM25Retriever.from_documents(
            documents, k=retrieved_document_count
        )

        embeddings = HuggingFaceEmbeddings(show_progress=True)
        if os.path.exists(arguments.persist_directory):
            self.faiss_vectorstore = FAISS.load_local(
                arguments.persist_directory,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self.faiss_vectorstore = FAISS.from_documents(
                documents,
                embeddings,
            )
            self.faiss_vectorstore.save_local(
                arguments.persist_directory
            )
        self.faiss_retriever = (
            self.faiss_vectorstore.as_retriever(
                search_kwargs={"k": retrieved_document_count}
            )
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
        query,
    ):
        return self.bm25_retriever.invoke(query)

    def search_semantic(
        self,
        query,
    ):
        return self.faiss_retriever.invoke(query)

    def search_hybrid(
        self,
        query,
    ):
        return self.ensemble_retriever.invoke(query)


def main():
    parser = argparse.ArgumentParser(
        prog="hybrid_searcher",
        description="Performs hybrid search with the given query",
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        help="Path of the input dataset (CSV file)",
        default=os.path.join("data", "imdb_dataset.csv"),
    )
    parser.add_argument(
        "-q",
        "--query_file_path",
        help="Path of the file that includes queries",
        default=os.path.join("data", "queries.json"),
    )
    parser.add_argument(
        "-p",
        "--persist_directory",
        help="Path of the persist directory",
        default=os.path.join("storage"),
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        help="Path of the output directory",
        default=os.path.join("output"),
    )
    arguments = parser.parse_args()

    with open(arguments.query_file_path, "r") as query_file:
        queries = json.load(query_file)

    model = OllamaLLM(model="llama3", temperature=0)

    searcher = Searcher(arguments)
    function_list = [
        searcher.search_lexical,
        searcher.search_semantic,
        searcher.search_hybrid,
    ]
    if not os.path.exists(arguments.output_directory):
        os.mkdir(arguments.output_directory)
    for search_function in function_list:
        with open(
            os.path.join(
                arguments.output_directory,
                f"{search_function.__name__}.json",
            ),
            "w",
        ) as output_file:
            query_results = list()

            for query in queries:
                query_result = dict()

                query_result["query"] = query
                query_result["search_function"] = (
                    search_function.__name__
                )
                retrieved_documents = search_function(query)
                query_result["retrieved_documents"] = list()
                for document in retrieved_documents:
                    prompt = (
                        """For the following query and retrieved document pair,
                        generate an evaluation score between 1 and 10 (inclusive)
                        that shows the relevance of the document to the query.
                        Query: """
                        + query_result["query"]
                        + """
                        Document: """
                        + document.page_content
                        + """
                        Your answers should be in this format (without any extra text):
                            {"score": "your score here ...", "reasoning": "your reasoning here ..." (string)}"""
                    )
                    llm_answer = model.invoke(prompt)
                    could_decode_json = False
                    while not could_decode_json:
                        try:
                            llm_answer_dictionary = json.loads(
                                llm_answer
                            )
                            could_decode_json = True
                        except (
                            json.decoder.JSONDecodeError
                        ) as error:
                            termcolor.cprint(
                                "Could not decode JSON", "magenta"
                            )
                            prompt = (
                                """Could not decode JSON, change this to a decodable JSON.
                                        Do not include anything else in your answer than JSON):
                                """
                                + llm_answer
                            )
                            llm_answer = model.invoke(prompt)

                    new_document = {
                        "id": document.metadata["row"],
                        "text": document.page_content,
                        "evaluation": json.loads(llm_answer),
                    }
                    query_result["retrieved_documents"].append(
                        new_document
                    )
                query_results.append(query_result)

            print(
                json.dumps(query_results, indent=4),
                file=output_file,
            )


if __name__ == "__main__":
    main()
