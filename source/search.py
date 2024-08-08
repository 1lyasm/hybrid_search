import llama_index
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)
from llama_index.llms.huggingface import HuggingFaceLLM
import termcolor


def do_semantic_search(query_engine, query_string):
    return query_engine.query(query_string)


def do_keyword_search(query_string):
    pass


def do_hybrid_search(query_string):
    pass


def main():
    termcolor.cprint("Loading vector store index", "blue")
    storage_context = StorageContext.from_defaults(
        persist_dir="storage"
    )
    index = llama_index.core.load_index_from_storage(
        storage_context,
        embed_model=HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        ),
    )

    termcolor.cprint("Performing searches", "blue")
    query_engine = index.as_query_engine(
        llm=HuggingFaceLLM(
            model_name="HuggingFaceH4/zephyr-7b-beta",
            tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
            context_window=3900,
            max_new_tokens=256,
            device_map="auto",
        )
    )

    print(do_semantic_search(query_engine, "Shakespeare"))


if __name__ == "__main__":
    main()
