# Query Intent Parsing
def parse_query_intent(query):
    q = query.lower()

    intent = {
        "raw_query": query,
        "buyer_contains": None,
        "supplier_contains": None,
        "category_terms": [],
        "region_terms": [],
        "cpv_code": None,
        "framework_required": False,
        "multi_supplier_required": False
    }

    if "nhs" in q:
        intent["buyer_contains"] = "nhs"

    if "london" in q:
        intent["region_terms"].extend(["london", "uki"])

    if "scotland" in q:
        intent["region_terms"].extend(["scotland", "ukm"])

    if "wales" in q:
        intent["region_terms"].extend(["wales", "ukl"])

    if "construction" in q:
        intent["category_terms"].extend(["construction", "works", "building"])

    if "it" in q or "technology" in q or "software" in q:
        intent["category_terms"].extend(["it", "technology", "software", "computer", "digital"])

    if "cloud" in q:
        intent["category_terms"].extend(["cloud", "hosting", "saas", "software"])

    if "framework" in q:
        intent["framework_required"] = True

    if "multi supplier" in q or "multi-supplier" in q:
        intent["multi_supplier_required"] = True

    cpv_match = re.search(r"\b\d{8}\b", q)
    if cpv_match:
        intent["cpv_code"] = cpv_match.group(0)

    supplier_match = re.search(r"awarded to ([a-zA-Z0-9 &.-]+)", q)
    if supplier_match:
        intent["supplier_contains"] = supplier_match.group(1).strip()

    return intent


# Metadata / Intent Pre-filtering
def apply_intent_prefilter(df, intent):
    filtered = df.copy()

    combined_text = (
        filtered["rag_text"]
        .fillna("")
        .str.lower()
    )

    if intent["buyer_contains"]:
        filtered = filtered[
            filtered["buyer_name"]
            .fillna("")
            .str.lower()
            .str.contains(intent["buyer_contains"], na=False)
        ]

    if intent["supplier_contains"]:
        filtered = filtered[
            filtered["supplier_names"]
            .fillna("")
            .str.lower()
            .str.contains(intent["supplier_contains"], na=False)
        ]

    if intent["cpv_code"]:
        filtered = filtered[
            filtered["cpv_codes"]
            .fillna("")
            .str.contains(intent["cpv_code"], na=False)
        ]

    if intent["category_terms"]:
        pattern = "|".join(intent["category_terms"])
        filtered = filtered[
            filtered["rag_text"]
            .fillna("")
            .str.lower()
            .str.contains(pattern, na=False)
        ]

    if intent["region_terms"]:
        pattern = "|".join(intent["region_terms"])
        filtered = filtered[
            filtered["rag_text"]
            .fillna("")
            .str.lower()
            .str.contains(pattern, na=False)
        ]

    if intent["framework_required"]:
        filtered = filtered[
            filtered["rag_text"]
            .fillna("")
            .str.lower()
            .str.contains("framework", na=False)
        ]

    if intent["multi_supplier_required"]:
        filtered = filtered[
            filtered["supplier_names"]
            .fillna("")
            .str.contains(",", na=False)
        ]

    # Fallback: if filters are too strict, return full dataset
    if len(filtered) < 5:
        return df.copy()

    return filtered
