# rag_llm_explainer.py
"""
RAG + LLM explainer scaffold for single-store analysis:
 - Builds a small local 'knowledge' store (CSV-based) for S01.
 - Retrieves context with a naive keyword search.
 - Calls OpenAI (or returns a mock response if API key is missing / invalid / quota exceeded).
"""

import os
import pandas as pd
import textwrap
from openai import OpenAI
import openai

DATA_DIR = "data"
KNOW_DIR = "knowledge"
os.makedirs(KNOW_DIR, exist_ok=True)

def build_knowledge_base():
    """Builds a toy knowledge base from daily store aggregates (last 60 days)."""
    agg = pd.read_csv(os.path.join(DATA_DIR, "daily_store_agg.csv"), parse_dates=["date"])
    agg["date"] = pd.to_datetime(agg["date"], errors="coerce")

    latest = agg[agg["date"] >= (agg["date"].max() - pd.Timedelta(days=60))]

    store_id = "S01"
    df = latest[latest["store_id"] == store_id]

    if df.empty:
        doc = f"Store {store_id} summary (last 60 days): no recent data available."
    else:
        mean_units = df["units_sold"].mean()
        median_price = df["price"].median()
        promo_days = df["on_promo"].sum()
        stockouts = df["stockout"].sum()

        doc = (
            f"Store {store_id} summary (last 60 days): "
            f"mean_units={mean_units:.1f}, "
            f"median_price={median_price:.2f}, "
            f"promo_days={promo_days}, "
            f"stockouts={stockouts}"
        )

    kb = pd.DataFrame([{"id": f"{store_id}_summary", "text": doc}])
    kb.to_csv(os.path.join(KNOW_DIR, "kb.csv"), index=False)
    print("KB built to", os.path.join(KNOW_DIR, "kb.csv"))

def retrieve(query, top_k=3):
    """Naive keyword-based retriever from the CSV knowledge base."""
    kb_path = os.path.join(KNOW_DIR, "kb.csv")
    if not os.path.exists(kb_path):
        return []

    kb = pd.read_csv(kb_path)
    kb["score"] = kb["text"].apply(
        lambda t: sum(1 for w in query.lower().split() if w.strip(".,?!") in t.lower())
    )
    out = kb.sort_values("score", ascending=False).head(top_k)
    return out["text"].tolist()

def call_openai_with_context(prompt, context, model="gpt-3.5-turbo"):
    """
    Wrapper to call OpenAI. Returns mock response if API key is missing or quota exceeded.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return f"[MOCK RESPONSE] No API key set. Based on context ({' | '.join(context)}), " \
               f"store performance may have been affected by pricing, promotions, or stock levels. " \
               f"Suggested action: review discount depth, promo timing, and inventory."

    client = OpenAI()  # picks up OPENAI_API_KEY automatically
    system = "You are a helpful retail operations assistant. Be concise and include actionable steps."
    combined = "\n\n".join(context) + "\n\n" + prompt

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": combined},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        return resp.choices[0].message.content

    except openai.RateLimitError:
        return f"[MOCK RESPONSE] Out of quota. Based on context ({' | '.join(context)}), " \
               f"review discount depth, promo timing, and inventory for '{prompt}'."

    except openai.AuthenticationError:
        return f"[MOCK RESPONSE] API key invalid. Context ({' | '.join(context)}). " \
               f"Suggested action: check competitor activity and stock levels."

    except Exception as e:
        return f"[MOCK RESPONSE due to OpenAI error: {e}]. Context ({' | '.join(context)}). " \
               f"Suggested action: review inventory and promotions."

if __name__ == "__main__":
    build_knowledge_base()
    query = "Why did store S01 see a drop in units last week and what should we do?"
    ctx = retrieve(query)
    print("Retrieved context:", ctx)
    print("------")
    print("Example prompt to LLM (real API if credits available, else mock response):")
    print(textwrap.fill("Context: " + " || ".join(ctx) + "   Question: " + query, width=120))

    out = call_openai_with_context(query, ctx)
    print(out)
