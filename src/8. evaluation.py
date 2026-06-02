assessment_eval_queries = [
    "Find framework agreements for IT services in the NHS",
    "Contracts awarded in London for construction",
    "Multi-supplier frameworks for cloud services"
]

manual_eval_rows = []

for query in assessment_eval_queries:
    results = rag_search_v2(query, top_k=10)

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        manual_eval_rows.append({
            "query": query,
            "rank": rank,
            "notice_id": row["Notice ID"],
            "title": row["Title"],
            "buyer": row["Buyer"],
            "suppliers": row["Suppliers"],
            "summary": row["Summary"],
            "relevance_score": row["Relevance Score"],
            "hybrid_score": row["Hybrid Score"],
            "manual_relevance": ""  # Fill manually: 2, 1, or 0
        })

manual_eval_df = pd.DataFrame(manual_eval_rows)

manual_eval_df

manual_eval_df.to_csv(
    "assessment_manual_evaluation_template.csv",
    index=False
)

manual_eval = pd.read_csv("assessment_manual_evaluation_filled.csv")
manual_eval["manual_relevance"] = manual_eval["manual_relevance"].astype(int)

manual_eval.head()


#Relaxed (1 or 2 counts)
def precision_at_k(group, k=5, threshold=1):

    top_k = (
        group
        .sort_values("rank")
        .head(k)
    )

    return (
        top_k["manual_relevance"] >= threshold
    ).mean()

#Strict (only 2 counts)
def precision_at_k_strict(group, k=5):

    top_k = (
        group
        .sort_values("rank")
        .head(k)
    )

    return (
        top_k["manual_relevance"] == 2
    ).mean()

def mrr(group):

    ranked = (
        group
        .sort_values("rank")
    )

    for rank, rel in enumerate(
        ranked["manual_relevance"],
        start=1
    ):

        if rel == 2:
            return 1 / rank

    return 0


#NDCG@5
def dcg(relevances):

    relevances = np.array(relevances)

    return np.sum(
        (2**relevances - 1)
        /
        np.log2(
            np.arange(
                2,
                len(relevances) + 2
            )
        )
    )


def ndcg_at_5(group):

    labels = (
        group
        .sort_values("rank")
        .head(5)["manual_relevance"]
        .tolist()
    )

    ideal_labels = sorted(
        labels,
        reverse=True
    )

    ideal_dcg = dcg(
        ideal_labels
    )

    if ideal_dcg == 0:
        return 0

    return dcg(labels) / ideal_dcg

results = []

for query, group in manual_eval.groupby("query"):

    results.append({
        "query": query,
        "precision_at_5_relaxed":
            precision_at_k(
                group,
                k=5,
                threshold=1
            ),

        "precision_at_5_strict":
            precision_at_k_strict(
                group,
                k=5
            ),

        "mrr":
            mrr(group),

        "ndcg_at_5":
            ndcg_at_5(group)
    })

metrics_df = pd.DataFrame(results)

metrics_df

print("\nOverall metrics:")
print(metrics_df.mean(numeric_only=True))

print("Per Query Evaluation Results")
display(metrics_df)
