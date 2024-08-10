import llama_index
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)
from llama_index.llms.huggingface import HuggingFaceLLM
import argparse


class SemanticSearcher:
    def __init__(self):
        storage_context = StorageContext.from_defaults(
            persist_dir="storage"
        )
        index = llama_index.core.load_index_from_storage(
            storage_context,
            embed_model=HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            ),
        )
        self.query_engine = index.as_query_engine(
            llm=HuggingFaceLLM(
                model_name="HuggingFaceH4/zephyr-7b-beta",
                tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
                context_window=3900,
                max_new_tokens=256,
                device_map="auto",
            )
        )

    def search(self, query):
        return self.query_engine.query(query)


def main():
    parser = argparse.ArgumentParser(
        prog="semantic_searcher",
        description="Performs semantic search with the given query",
    )
    parser.add_argument(
        "-q", "--query_string", help="Query string", required=True
    )
    arguments = parser.parse_args()

    searcher = SemanticSearcher()
    searcher.search(arguments.query_string)


if __name__ == "__main__":
    main()
