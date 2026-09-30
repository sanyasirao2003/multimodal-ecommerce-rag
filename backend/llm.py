import requests
from functools import lru_cache

OLLAMA_URL = "http://localhost:11434/api/generate"

# 🔥 FASTER MODEL
MODEL_NAME = "phi3"


# 🔥 CACHE (instant repeat responses)
@lru_cache(maxsize=500)
def generate_explanation_cached(query, title, brand, price, category):

# 🔥 SHORT PROMPT (VERY IMPORTANT)
    prompt = f"""
You are a smart shopping assistant.

User query: {query}

Product:
Title: {title}
Brand: {brand}
Price: ₹{price}
Category: {category}

Explain why this is the best choice.

Rules:
- Write EXACTLY 3 sentences
- Each sentence MUST be on a new line
- Mention category relevance
- Mention price advantage

Answer:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 180,   # 🔥 LIMIT TOKENS (SPEED BOOST)
                    "temperature": 0.7
                }
            },
            timeout=10,
        )

        data = response.json()

        if "response" in data:
            return data["response"].strip()

        return "Good match based on your search."

    except Exception:
        return "This product matches your requirements well."


# 🔥 MAIN FUNCTION
def generate_explanation(query, best_product):

    return generate_explanation_cached(
        query,
        best_product["title"],
        best_product["brand"],
        best_product["price"],
        best_product["category"]
    )

def generate_suggestions(query, products):

    context = ""
    for p in products[:2]:
        context += f"""
Title: {p['title']}
Category: {p['category']}
Price: ₹{p['price']}
"""

    prompt = f"""
User query: {query}

Products:
{context}

Generate 4 short follow-up questions.
Rules:
- Max 6 words
- No numbering
- One per line
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 60}
            },
            timeout=10,
        )

        data = response.json()
        text = data.get("response", "")

        suggestions = [
            line.strip("- ").strip()
            for line in text.split("\n")
            if line.strip()
        ]

        return suggestions[:4]

    except Exception:
        return [
            "Which is better?",
            "Best under budget?",
            "Good for daily use?",
            "Any cheaper option?"
        ]