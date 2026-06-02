# Data Cleaning and Normalisation
import numpy as np
import pandas as pd

# 1. Replace empty strings with NaN
df_contracts = df_contracts.replace("", np.nan)

# 2. Clean whitespace in text columns
text_columns = [
    "title",
    "description",
    "buyer_name",
    "supplier_names",
    "cpv_codes",
    "cpv_descriptions",
    "regions",
    "tender_status",
    "award_statuses",
    "contract_statuses",
    "procurement_method",
    "procurement_method_details",
    "main_procurement_category",
    "notice_url",
    "search_text"
]

for col in text_columns:
    if col in df_contracts.columns:
        df_contracts[col] = (
            df_contracts[col]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

# 3. Convert date column
df_contracts["date"] = pd.to_datetime(df_contracts["date"], errors="coerce", utc=True)

# 4. Convert contract value column
df_contracts["max_contract_value"] = pd.to_numeric(
    df_contracts["max_contract_value"],
    errors="coerce"
)

# 5. Add useful boolean flags
df_contracts["has_supplier"] = df_contracts["supplier_names"].notna()
df_contracts["has_cpv"] = df_contracts["cpv_codes"].notna()
df_contracts["has_region"] = df_contracts["regions"].notna()
df_contracts["has_contract_value"] = df_contracts["max_contract_value"].notna()

# 6. Create contract year
df_contracts["year"] = df_contracts["date"].dt.year

print(df_contracts.shape)
df_contracts.head()

# Build Structured Search Text
def build_structured_search_text(row):
    parts = []

    if pd.notna(row.get("title")):
        parts.append(f"Title: {row['title']}")

    if pd.notna(row.get("description")):
        parts.append(f"Description: {row['description']}")

    if pd.notna(row.get("buyer_name")):
        parts.append(f"Buyer: {row['buyer_name']}")

    if pd.notna(row.get("supplier_names")):
        parts.append(f"Suppliers: {row['supplier_names']}")

    if pd.notna(row.get("cpv_descriptions")):
        parts.append(f"CPV category: {row['cpv_descriptions']}")

    if pd.notna(row.get("cpv_codes")):
        parts.append(f"CPV code: {row['cpv_codes']}")

    if pd.notna(row.get("regions")):
        parts.append(f"Region: {row['regions']}")

    if pd.notna(row.get("procurement_method")):
        parts.append(f"Procurement method: {row['procurement_method']}")

    if pd.notna(row.get("procurement_method_details")):
        parts.append(f"Procurement method details: {row['procurement_method_details']}")

    if pd.notna(row.get("main_procurement_category")):
        parts.append(f"Main procurement category: {row['main_procurement_category']}")

    if pd.notna(row.get("max_contract_value")):
        parts.append(f"Maximum contract value GBP: {row['max_contract_value']}")

    return "\n".join(parts)


df_contracts["search_text"] = df_contracts.apply(build_structured_search_text, axis=1)

# Sanity Check
print("Total rows:", len(df_contracts))
print("Unique OCIDs:", df_contracts["ocid"].nunique())
print("Duplicate OCIDs:", df_contracts["ocid"].duplicated().sum())
print("Unique Notice IDs:", df_contracts["notice_id"].nunique())

# Save Final Retrieval Dataset
# one row per notice/contract record.
# The dataset already has unique OCIDs and notice IDs, so no deduplication is required.

df_retrieval = df_contracts.copy()

# Droping list-style column because it is not required for retrieval.
if "contract_values" in df_retrieval.columns:
    df_retrieval = df_retrieval.drop(columns=["contract_values"])

df_retrieval.to_csv("final_retrieval_contracts_2026.csv", index=False)

print(df_retrieval.shape)
df_retrieval.head()

# Build RAG Text
def safe_text(value):
    return "" if value is None or pd.isna(value) else str(value)

def build_rag_text(row):
    return f"""
Title: {safe_text(row.get('title'))}
Description: {safe_text(row.get('description'))}
Buyer: {safe_text(row.get('buyer_name'))}
Suppliers: {safe_text(row.get('supplier_names'))}
CPV Codes: {safe_text(row.get('cpv_codes'))}
CPV Categories: {safe_text(row.get('cpv_descriptions'))}
Regions: {safe_text(row.get('regions'))}
Procurement Method: {safe_text(row.get('procurement_method'))}
Procurement Details: {safe_text(row.get('procurement_method_details'))}
Main Category: {safe_text(row.get('main_procurement_category'))}
Value GBP: {safe_text(row.get('max_contract_value'))}
""".strip()

df["rag_text"] = df.apply(build_rag_text, axis=1)
