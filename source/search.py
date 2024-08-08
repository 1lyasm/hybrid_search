import llama_index
from llama_index.core.storage import StorageContext
import llama_index.embeddings.huggingface as huggingface
import termcolor


def main():
    termcolor.cprint("Loading vector store index", "blue")
    storage_context = StorageContext.from_defaults(
        persist_dir="storage"
    )

    index = llama_index.core.load_index_from_storage(
        storage_context,
        embed_model=huggingface.HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        ),
    )


if __name__ == "__main__":
    main()
