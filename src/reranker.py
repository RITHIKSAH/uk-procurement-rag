from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query, results, top_k=5):
    pairs = []

    for _, row in results.iterrows():
        pairs.append([
            query,
            row["rag_text"]
        ])

    rerank_scores = reranker.predict(pairs)

    results = results.copy()
    results["rerank_score"] = rerank_scores

    results = results.sort_values(
        "rerank_score",
        ascending=False
    )

    return results.head(top_k)

def hybrid_retrieve_with_reranking(query, top_k=5, candidate_k=100):
    results, intent, vector_ids, bm25_ids = hybrid_retrieve(
        query=query,
        top_k=candidate_k,
        candidate_k=candidate_k
    )

    reranked_results = rerank_results(
        query=query,
        results=results,
        top_k=top_k
    )

    return reranked_results, intent

def explain_match_v2(row, query, intent):
    matched_terms = keyword_overlap(query, row["rag_text"])

    explanation_parts = []

    explanation_parts.append(
        f"Semantic + BM25 candidate retrieval followed by cross-encoder re-ranking."
    )

    explanation_parts.append(
        f"Cross-encoder relevance score: {row['rerank_score']:.3f}"
    )

    if row.get("rrf_score") is not None:
        explanation_parts.append(
            f"Hybrid RRF score: {row['rrf_score']:.4f}"
        )

    if matched_terms:
        explanation_parts.append(
            "Keyword overlap: " + ", ".join(matched_terms[:10])
        )

    if intent.get("buyer_contains"):
        explanation_parts.append(
            f"Buyer intent matched/considered: {intent['buyer_contains']}"
        )

    if intent.get("category_terms"):
        explanation_parts.append(
            "Category intent considered: " + ", ".join(intent["category_terms"])
        )

    if intent.get("region_terms"):
        explanation_parts.append(
            "Region intent considered: " + ", ".join(intent["region_terms"])
        )

    if intent.get("framework_required"):
        explanation_parts.append(
            "Query requested framework-related contracts."
        )

    if intent.get("multi_supplier_required"):
        explanation_parts.append(
            "Query requested multi-supplier contracts/frameworks."
        )

    if pd.notna(row.get("cpv_descriptions")):
        explanation_parts.append(
            f"CPV context: {row['cpv_descriptions']}"
        )

    return " | ".join(explanation_parts)

def rag_search_v2(query, top_k=5):
    results, intent = hybrid_retrieve_with_reranking(
        query=query,
        top_k=top_k,
        candidate_k=150
    )

    output = []

    for _, row in results.iterrows():
        output.append({
            "Notice ID": row.get("notice_id"),
            "Title": row.get("title"),
            "Buyer": row.get("buyer_name"),
            "Suppliers": row.get("supplier_names"),
            "Summary": extractive_summary(row.get("description"), max_words=80),
            "Relevance Score": round(float(row.get("rerank_score", 0)), 4),
            "Hybrid Score": round(float(row.get("rrf_score", 0)), 4),
            "Why it matched": explain_match_v2(row, query, intent)
        })

    return pd.DataFrame(output)
