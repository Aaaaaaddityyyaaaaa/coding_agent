from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

user_name = "user123"

lc_embedding_fn = HuggingFaceEmbeddings(
    model_name="flax-sentence-embeddings/st-codesearch-distilroberta-base"
)

store = Chroma(
    persist_directory="./chroma",
    collection_name=f"{user_name}documents",
    embedding_function=lc_embedding_fn
)

# Switched from jina-reranker-v2 (broken with current transformers) to a
# reliable cross-encoder that needs no custom code and works out of the box.
re_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retriever(query: str) -> list[tuple[str, dict, float]]:
    """Retrieves and reranks the most relevant chunks from ChromaDB.
    Returns list of (doc_text, metadata, score) tuples, sorted best-first.
    """
    results = store.similarity_search(query=query, k=10)

    docs      = [r.page_content for r in results]
    metadatas = [r.metadata     for r in results]

    pairs  = [[query, doc] for doc in docs]
    scores = re_ranker.predict(pairs)

    reranked = sorted(
        zip(docs, metadatas, scores),
        key=lambda x: x[2],
        reverse=True
    )[:5]

    return reranked