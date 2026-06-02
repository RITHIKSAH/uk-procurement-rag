# BM25 Keyword retrieval

import re
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", str(text).lower())

tokenized_corpus = [tokenize(text) for text in df["rag_text"].tolist()]

bm25 = BM25Okapi(tokenized_corpus)

# Hybrid Retrieval with Reciprocal Rank Fusion
def reciprocal_rank_fusion(rank_lists, k=60):
    scores = {}

    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_retrieve(query, top_k=5, candidate_k=50):
    intent = parse_query_intent(query)

    filtered_df = apply_intent_prefilter(df, intent)

    allowed_ids = set(filtered_df["notice_id"].astype(str).tolist())

    # Vector search
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()[0]

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k
    )

    vector_ids = vector_results["ids"][0]
    vector_distances = vector_results["distances"][0]

    vector_ranked_ids = [
        doc_id for doc_id in vector_ids
        if doc_id in allowed_ids
    ]

    # BM25 search
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_ranked_indices = np.argsort(bm25_scores)[::-1][:candidate_k * 3]

    bm25_ranked_ids = [
        str(df.iloc[i]["notice_id"])
        for i in bm25_ranked_indices
        if str(df.iloc[i]["notice_id"]) in allowed_ids
    ][:candidate_k]

    # RRF fusion
    fused = reciprocal_rank_fusion(
        [vector_ranked_ids, bm25_ranked_ids]
    )

    final_ids = [doc_id for doc_id, score in fused[:top_k]]
    fusion_scores = dict(fused)

    results = df[
        df["notice_id"].astype(str).isin(final_ids)
    ].copy()

    results["rrf_score"] = results["notice_id"].astype(str).map(fusion_scores)

    results = results.sort_values("rrf_score", ascending=False)

    return results, intent, vector_ranked_ids, bm25_ranked_ids


# Result Summaries and Match Explanations
def extractive_summary(text, max_words=70):
    words = str(text).split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

def keyword_overlap(query, text):
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))

    stopwords = {
        "the", "and", "for", "with", "from", "find", "show",
        "contract", "contracts", "service", "services", "awarded"
    }

    query_terms = query_terms - stopwords

    matched = sorted(query_terms.intersection(text_terms))

    return matched

def explain_match(row, query, intent):
    matched_terms = keyword_overlap(query, row["rag_text"])

    explanations = []

    explanations.append("Retrieved using hybrid semantic vector search and BM25 keyword search.")

    if matched_terms:
        explanations.append(
            "Keyword overlap: " + ", ".join(matched_terms[:10])
        )

    if intent["buyer_contains"]:
        explanations.append(
            f"Buyer filter/context matched: {intent['buyer_contains']}"
        )

    if intent["category_terms"]:
        explanations.append(
            "Category intent terms considered: " + ", ".join(intent["category_terms"])
        )

    if intent["region_terms"]:
        explanations.append(
            "Region intent terms considered: " + ", ".join(intent["region_terms"])
        )

    if intent["framework_required"]:
        explanations.append("Framework requirement identified in query.")

    if intent["multi_supplier_required"]:
        explanations.append("Multi-supplier requirement identified in query.")

    if row.get("cpv_descriptions"):
        explanations.append(
            f"CPV context: {row.get('cpv_descriptions')}"
        )

    return " | ".join(explanations)
