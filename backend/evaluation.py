def precision_at_k(products, query, k=5):

    query_lower = query.lower()
    relevant = 0

    for product in products[:k]:

        title = product["title"].lower()
        category = product["category"].lower()

        # Basic relevance rule
        if "formal" in query_lower:
            if "formal" in category:
                relevant += 1
        else:
            if any(word in title for word in query_lower.split()):
                relevant += 1

    return relevant / k