import llama_index
import llama_index.embeddings.huggingface as huggingface


def main():
    llama_index.core.Settings.embed_model = huggingface.HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    documents = llama_index.core.SimpleDirectoryReader("./data").load_data()
    index = llama_index.core.VectorStoreIndex.from_documents(
        documents, show_progress=True
    )
    storage_context = llama_index.core.storage.StorageContext.from_defaults(index_store=index)
    storage_context.persist()

if __name__ == "__main__":
    main()
