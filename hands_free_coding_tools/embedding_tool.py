import chromadb
from chromadb.utils import embedding_functions

user_name = "user123"

client = chromadb.PersistentClient(path="./chroma")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="flax-sentence-embeddings/st-codesearch-distilroberta-base"
)

collection = client.get_or_create_collection(
    name=f"{user_name}documents",
    embedding_function=embedding_fn
)


def embed_chunks(python_chunks: list, java_chunks: list, js_chunks: list, c_chunks: list):
    """Embeds all language chunks into the ChromaDB collection."""
    all_chunks = python_chunks + java_chunks + js_chunks + c_chunks
    for chunk in all_chunks:
        collection.add(
            ids=[f"{chunk['file_path']}_{chunk['name']}_{chunk['start_line']}"],
            documents=[chunk["code"]],
            metadatas=[{
                "name":       chunk["name"],
                "type":       chunk["type"],
                "file_path":  chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line":   chunk["end_line"],
            }]
        )