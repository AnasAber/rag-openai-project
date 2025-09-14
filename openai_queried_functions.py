from datetime import datetime
from utils import ai_use_log
from openai import OpenAI
from typing import List
import json
import os


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=OPENAI_API_KEY)


def log_ai_use(action, details):
    """Log AI usage with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ai_use_log.append({
        'timestamp': timestamp,
        'action': action,
        'details': details
    })
    print(f"[LOG] {timestamp}: {action}")



def generate_answer(query, store, top_k: int, top_n: int):
    # Retrieve context
    results = store.search(query, top_k=top_k)


    # this function logs activities to ai_use_log, it will later be saved in ai_use_log.json
    log_ai_use("retrieval", f"Top-{top_k} docs for query '{query}'")

    reranked_context = rerank_with_model(query, results, top_n=top_n)

    log_ai_use("reranking and retrieval of", f"Top-{top_n} docs for query '{query}'")

    prompt = f"""
    You are a precise assistant. Answer strictly from the provided CONTEXT.
    Context:
    {reranked_context}

    Question: {query}

    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cuz it's cheaper
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    log_ai_use("generation", f"Answer generated for query '{query}'")

    return response.choices[0].message.content


# Reranking: Improve retrieval results

def rerank_with_model(query: str, candidates: list, top_n=3) -> list:
    """
    Using gpt-4o-mini to score candidates for relevance to the query (0-100).
    Return the top_n most relevant.
    """
    # Build a simpler prompt
    prompt = f"""
    Query: {query}

    Candidate snippets:
    { [c[:200] for c in candidates] }  # only show first 200 chars per snippet

    Task: For each snippet, give a relevance score 0-100 as a JSON list.
    Example: [80, 10, 65]
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )

    text = response.choices[0].message.content.strip()

    # Try to parse model output into scores
    try:
        scores = json.loads(text)
    except:
        scores = [50] * len(candidates)  # fallback

    # Pair and sort
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_n]]