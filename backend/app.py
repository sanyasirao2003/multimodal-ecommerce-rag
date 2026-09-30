from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import os
import random
import requests
from backend.retrieval import retrieve_products
from backend.llm import generate_explanation
from backend.evaluation import precision_at_k
from fastapi import UploadFile, File, Form
from backend.image_retrieval import retrieve_hybrid
from deep_translator import GoogleTranslator
from langdetect import detect
from collections import defaultdict
from datetime import datetime, timedelta

# -------------------------
# INIT APP
# -------------------------
app = FastAPI()

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# SERVE IMAGES
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# =========================
# SESSION MANAGEMENT
# Per-session chat history stored in memory.
# Each session is identified by a session_id passed from the frontend.
# Sessions expire after 1 hour of inactivity.
# =========================

chat_sessions: dict[str, list[str]] = defaultdict(list)
session_last_accessed: dict[str, datetime] = {}

SESSION_TTL = timedelta(hours=1)

def get_session(session_id: str) -> list[str]:
    """Return history for this session and refresh its TTL."""
    now = datetime.now()

    # Lazy cleanup: evict expired sessions on every access
    expired = [
        sid for sid, ts in session_last_accessed.items()
        if now - ts > SESSION_TTL
    ]
    for sid in expired:
        chat_sessions.pop(sid, None)
        session_last_accessed.pop(sid, None)

    session_last_accessed[session_id] = now
    return chat_sessions[session_id]


def push_history(history: list[str], role: str, text: str, max_turns: int = 10):
    """Append a message and keep only the last max_turns lines."""
    history.append(f"{role}: {text}")
    if len(history) > max_turns:
        del history[: len(history) - max_turns]


# =========================
# SUGGESTION ENGINE
# Generates dynamic, query-aware chips that change every call.
# =========================

# Pool of suggestion templates per category keyword
SUGGESTION_POOLS = {
    "shoe": [
        "Best running shoes under ₹1000",
        "Waterproof shoes for men",
        "Lightweight sports shoes",
        "Casual sneakers for daily use",
        "Top rated white shoes",
        "Slip-on shoes for women",
        "Budget shoes under ₹500",
        "Formal leather shoes",
    ],
    "shirt": [
        "Slim fit cotton shirts",
        "Formal shirts under ₹800",
        "Half sleeve shirts for summer",
        "Best rated polo shirts",
        "White shirts for office",
        "Oversized shirts for men",
        "Printed shirts under ₹600",
        "Linen shirts for men",
    ],
    "jean": [
        "Slim fit jeans under ₹700",
        "Stretchable black jeans",
        "Ripped jeans for men",
        "High waist jeans for women",
        "Dark blue denim jeans",
        "Straight fit jeans",
        "Best rated jeans under ₹900",
        "Jogger jeans for casual wear",
    ],
    "pant": [
        "Formal trousers under ₹800",
        "Chinos for men",
        "Track pants for workout",
        "Cargo pants with pockets",
        "Slim fit trousers",
        "Linen pants for summer",
    ],
    "sunglass": [
        "Polarized sunglasses under ₹500",
        "UV400 protection glasses",
        "Aviator sunglasses for men",
        "Cat eye sunglasses for women",
        "Sports sunglasses",
        "Wayfarer style sunglasses",
    ],
    "jacket": [
        "Bomber jacket for men",
        "Lightweight jackets under ₹1000",
        "Hooded jacket for winter",
        "Denim jacket for casual wear",
        "Waterproof jacket for rain",
        "Fleece jacket under ₹700",
    ],
    "watch": [
        "Best smartwatch under ₹2000",
        "Analog watches for men",
        "Waterproof watches",
        "Sports watches for gym",
        "Digital watches under ₹500",
        "Luxury watches for gifting",
    ],
    "bag": [
        "Backpack for college",
        "Laptop bag under ₹800",
        "Stylish handbags for women",
        "Gym bag with compartments",
        "Waterproof travel bag",
        "Mini crossbody bag",
    ],
}

# Generic fallback pool — randomized so they change on every call
GENERIC_POOL = [
    "Best products under ₹500",
    "Top rated items today",
    "New arrivals this week",
    "Bestsellers in clothing",
    "Compare prices",
    "Show me something cheaper",
    "What's trending now?",
    "Help me find a gift",
    "Best value for money",
    "Most reviewed products",
    "Products with 4+ rating",
    "Show deals today",
]


def build_suggestions(query: str, products: list[dict], n: int = 4) -> list[str]:
    """
    Build n dynamic suggestion chips relevant to the query.
    Pulls from category-specific pools first, then generic.
    Shuffles so suggestions change on every call (Rufus-style refresh).
    Also injects one product-title-based chip when products are available.
    """
    query_lower = query.lower()

    # Pick category pool
    pool = []
    for keyword, suggestions in SUGGESTION_POOLS.items():
        if keyword in query_lower:
            pool = suggestions[:]
            break

    if not pool:
        pool = GENERIC_POOL[:]

    random.shuffle(pool)
    chips = pool[: n - 1]  # reserve one slot for a product-based chip

    # Add a product-specific chip if we have results
    if products:
        best = products[0]
        brand = best.get("brand", "")
        category = best.get("category", "")
        price = best.get("price", 0)

        product_chips = [
            f"More {category} options",
            f"{brand} products",
            f"Similar under ₹{int(price) + 200}",
            f"Compare with alternatives",
        ]
        random.shuffle(product_chips)
        chips.append(product_chips[0])

    # Final shuffle so order is different too
    random.shuffle(chips)
    return chips[:n]


# =========================
# HELPER FUNCTIONS
# =========================

def normalize_query(query: str) -> str:
    """Transliterate common Hindi/regional shopping words to English."""
    if not query:
        return ""
    q = query.lower()
    replacements = {
        "kala": "black",
        "lal": "red",
        "safed": "white",
        "neela": "blue",
        "pant": "pants",
        "shirt": "shirt",
    }
    for k, v in replacements.items():
        q = q.replace(k, v)
    return q


def detect_language(text: str) -> str:
    try:
        return detect(text) if text else "en"
    except Exception:
        return "en"


def translate_to_english(text: str) -> str:
    try:
        if not text or detect_language(text) == "en":
            return text
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


def call_ollama(prompt: str, max_tokens: int = 150) -> str | None:
    """
    Call local Ollama (phi3). Returns text or None on failure.
    Timeout is short so the API stays fast even if Ollama is down.
    """
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.75},
            },
            timeout=8,
        )
        data = res.json()
        return data.get("response", "").strip() or None
    except Exception:
        return None


def build_product_context(products: list[dict]) -> str:
    lines = []
    for p in products[:3]:
        lines.append(
            f"• {p['title']} | Brand: {p['brand']} | "
            f"₹{p['price']} | Rating: {p.get('rating', 4.0)}/5"
        )
    return "\n".join(lines)


def smart_fallback_response(query: str, products: list[dict]) -> str:
    """
    Rule-based fallback when Ollama is unavailable.
    Produces a different response style each call so it doesn't feel stale.
    """
    if not products:
        return (
            f"I couldn't find products for '{query}' right now. "
            "Try rephrasing or browsing a different category!"
        )

    best = products[0]
    title = best["title"]
    price = best["price"]
    rating = best.get("rating", 4.0)
    brand = best.get("brand", "")

    templates = [
        f"🏆 My top pick for '{query}' is **{title}** by {brand} — "
        f"priced at ₹{price} with a {rating}/5 rating. Great value!",

        f"For '{query}', I'd suggest **{title}** (₹{price}). "
        f"It has a solid {rating}/5 rating and comes from {brand}. "
        "Would you like to see cheaper options or compare alternatives?",

        f"✨ Looking for {query}? **{title}** by {brand} is a top match — "
        f"₹{price} and rated {rating}/5 by customers.",

        f"Based on your search, **{title}** stands out at ₹{price} "
        f"with a {rating}/5 rating. It's one of the best in this category!",
    ]
    return random.choice(templates)


# =========================
# SEARCH API
# =========================

@app.get("/search")
def search(
    query: str = Query(..., min_length=2),
    sort: str = Query("relevance", pattern="^(relevance|price_low|price_high)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, le=50),
):
    query = normalize_query(query)
    lang = detect_language(query)
    translated_query = translate_to_english(query) if lang != "en" else query

    print("User Lang:", lang)
    print("Final Query:", translated_query)

    all_products = retrieve_products(translated_query, k=60)

    if not all_products:
        return {
            "query": query,
            "message": "No products found.",
            "recommended_products": [],
            "best_product": None,
            "explanation": None,
            "precision_at_5": 0.0,
            "current_page": 1,
            "total_pages": 1,
        }

    # Sorting
    if sort == "price_low":
        all_products = sorted(all_products, key=lambda x: x["price"])
    elif sort == "price_high":
        all_products = sorted(all_products, key=lambda x: x["price"], reverse=True)
    else:
        all_products = sorted(
            all_products, key=lambda x: x.get("similarity_score", 0), reverse=True
        )

    # Pagination
    total_results = len(all_products)
    total_pages = (total_results + page_size - 1) // page_size
    page = min(page, total_pages)
    start = (page - 1) * page_size
    recommended_products = all_products[start : start + page_size]

    best_product = sorted(
        all_products,
        key=lambda x: (-x["rating"], x["price"], -len(x.get("reviews", []))),
    )[0]

    try:
        explanation = generate_explanation(translated_query, best_product)
    except Exception:
        explanation = "Explanation currently unavailable."

    precision = precision_at_k(all_products, translated_query, k=5)

    return {
        "query": query,
        "translated_query": translated_query,
        "language": lang,
        "recommended_products": recommended_products,
        "best_product": best_product,
        "explanation": explanation,
        "precision_at_5": precision,
        "current_page": page,
        "total_pages": total_pages,
    }


# =========================
# CHAT API  — Rufus-style
# =========================

@app.get("/chat")
def chat(
    query: str = Query(..., min_length=1),
    session_id: str = Query(default="default"),
    search_context: str = Query(default=""),   # last searched query from frontend
):
    """
    Rufus-style chat endpoint.
    - Per-session history (isolated per browser tab / user)
    - Dynamic suggestions that shuffle every request
    - Graceful Ollama fallback — never returns an empty response
    - Aware of the current search context passed by the frontend
    """

    # --- Normalize + translate ---
    raw_query = normalize_query(query)
    lang = detect_language(raw_query)
    translated_query = translate_to_english(raw_query) if lang != "en" else raw_query
    q_lower = translated_query.lower()

    # --- Session history ---
    history = get_session(session_id)
    push_history(history, "User", translated_query)

    # --- Greeting handler ---
    greetings = {"hi", "hello", "hey", "namaste", "good morning", "good evening", "sup", "yo"}
    if q_lower in greetings:
        if search_context:
            response = (
                f"👋 Hi! I see you're looking at '{search_context}'. "
                "I'm ScoutAI — want me to help you pick the best one, "
                "find cheaper options, or compare brands?"
            )
        else:
            response = (
                "👋 Hello! I'm ScoutAI, your smart shopping assistant. "
                "Tell me what you're looking for and I'll find the best options for you!"
            )
        push_history(history, "ScoutAI", response)
        return {
            "response": response,
            "suggestions": build_suggestions(search_context or "general", [], n=4),
        }

    # --- Thanks handler ---
    thanks = {"thanks", "thank you", "thx", "ty", "thank"}
    if q_lower in thanks:
        response = "You're welcome! 😊 Happy to help anytime. Anything else you'd like to explore?"
        push_history(history, "ScoutAI", response)
        return {
            "response": response,
            "suggestions": build_suggestions(search_context or translated_query, [], n=4),
        }

    # --- Detect UI action chip clicks ---
    # Chips like "Compare with alternatives", "More about black shoes",
    # "Similar under ₹989", "Bestsellers in clothing" are NOT real product
    # queries. If we detect one, use search_context for product retrieval
    # so we never return something unrelated (like a microphone for shoes).
    UI_ACTION_PATTERNS = [
        "compare", "alternatives", "more about", "show me", "something cheaper",
        "another option", "cheaper option", "best rated", "show more",
        "show deals", "bestsellers", "products with", "similar under",
        "tell me more", "details", "help me", "browse", "new arrivals",
        "popular items", "trending", "any offers", "find products",
        "more options", "other options",
    ]
    is_ui_action = any(p in q_lower for p in UI_ACTION_PATTERNS)

    # Use search_context as the effective product query for chip actions
    effective_query = search_context if (is_ui_action and search_context) else translated_query

    # Strip "More about " prefix if the chip was "More about black shoes"
    if effective_query.lower().startswith("more about "):
        effective_query = effective_query[len("more about "):]

    # --- Fetch relevant products (always against effective_query) ---
    products = retrieve_products(effective_query, k=10)[:3]

    # --- Build Ollama prompt ---
    history_text = "\n".join(history[-8:])
    product_context = build_product_context(products)
    context_note = (
        f"The user is shopping for: '{search_context}'.\n" if search_context else ""
    )

    # Explain what the user's chip click meant in plain English for the LLM
    action_note = ""
    if is_ui_action and search_context:
        action_note = (
            f"The user clicked a suggestion chip saying '{translated_query}'. "
            f"Treat this as a follow-up about '{search_context}'.\n"
        )

    prompt = f"""You are ScoutAI, a concise and friendly e-commerce shopping assistant (like Amazon Rufus).

{context_note}{action_note}Conversation so far:
{history_text}

Top matching products for '{effective_query}':
{product_context if product_context else "No matching products found."}

Instructions:
- Recommend ONE specific product from the list by name
- Mention its price and rating
- Keep reply to 2-3 short sentences
- Be warm and conversational
- End with a short follow-up question to keep the conversation going

ScoutAI:"""

    # --- Try Ollama first, fall back to rule-based ---
    llm_response = call_ollama(prompt, max_tokens=150)
    response_text = llm_response if llm_response else smart_fallback_response(effective_query, products)

    push_history(history, "ScoutAI", response_text)

    # --- Build dynamic suggestions ---
    # Always seed from effective_query (resolves to search_context for chip clicks)
    # so suggestions stay on the actual product topic, not the chip phrase
    suggestion_seed = effective_query if effective_query else search_context
    suggestions = build_suggestions(suggestion_seed, products, n=4)

    return {
        "response": response_text,
        "suggestions": suggestions,
    }


# =========================
# CLEAR CHAT (per session)
# =========================

@app.delete("/chat/clear")
def clear_chat(session_id: str = Query(default="default")):
    chat_sessions.pop(session_id, None)
    session_last_accessed.pop(session_id, None)
    return {"message": "Chat history cleared", "session_id": session_id}


# =========================
# IMAGE SEARCH
# =========================

@app.post("/image-search")
async def image_search(
    image: UploadFile = File(None),
    query: str = Form(""),
):
    query = normalize_query(query)
    translated_query = translate_to_english(query) if query else ""

    results = retrieve_hybrid(image.file if image else None, translated_query)

    if not results:
        return {
            "recommended_products": [],
            "best_product": None,
            "explanation": "No products found",
            "current_page": 1,
            "total_pages": 1,
        }

    best_product = sorted(
        results,
        key=lambda x: (-x["rating"], x["price"], -len(x.get("reviews", []))),
    )[0]

    try:
        explanation = generate_explanation(translated_query, best_product)
    except Exception:
        explanation = "Explanation unavailable"

    return {
        "recommended_products": results,
        "best_product": best_product,
        "explanation": explanation,
        "current_page": 1,
        "total_pages": 1,
    }


# =========================
# INITIAL SUGGESTIONS API
# Returns randomized suggestions every call — feeds the chat open screen
# =========================

@app.get("/suggestions")
def get_suggestions():
    pool = [
        "Best shoes under ₹1000",
        "Black shirts for men",
        "Top rated sunglasses",
        "Casual wear for daily use",
        "Affordable running shoes",
        "Best clothing deals",
        "Slim fit jeans under ₹700",
        "Waterproof jackets",
        "Smartwatch under ₹2000",
        "Backpack for college",
        "Cotton t-shirts for summer",
        "Formal trousers under ₹800",
    ]
    random.shuffle(pool)
    return {"suggestions": pool[:6]}
