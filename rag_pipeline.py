from openai_queried_functions import generate_answer
from embedding_functions import ingest_documents


if __name__ == '__main__':
    store = ingest_documents()
    query = input(" > Ask a question: ")
    # top-k is for the initial retrieval, top-n is for the reranking
    answer = generate_answer(query, store, top_k=5, top_n=3)
    print("💡 Answer:", answer)

    # Save log at the end
    import json
    with open("logs_and_costs/ai_use_log.json", "a", encoding="utf-8") as f:
        from utils import ai_use_log
        json.dump(ai_use_log, f, indent=2)
    print(" *** Log saved to ai_use_log.json *** ")