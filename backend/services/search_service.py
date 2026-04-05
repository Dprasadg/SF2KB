from backend.search.rag import search_kb

def search(query, embedder, vector_store):
    return search_kb(query, embedder, vector_store)