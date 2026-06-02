import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="./chroma_fts_db")
collection = client.get_or_create_collection(
    name="fts_contracts"
)
print("Existing Chroma records:", collection.count())

documents = df["rag_text"].tolist()
ids = df["notice_id"].astype(str).tolist()

metadatas = []

for _, row in df.iterrows():
    metadatas.append({
        "notice_id": safe_text(row.get("notice_id")),
        "title": safe_text(row.get("title")),
        "buyer_name": safe_text(row.get("buyer_name")),
        "supplier_names": safe_text(row.get("supplier_names")),
        "cpv_codes": safe_text(row.get("cpv_codes")),
        "cpv_descriptions": safe_text(row.get("cpv_descriptions")),
        "regions": safe_text(row.get("regions")),
        "procurement_method": safe_text(row.get("procurement_method")),
        "max_contract_value": safe_text(row.get("max_contract_value"))
    })

batch_size = 500

# Only populate Chroma if the collection is empty.
# This prevents duplicate ID errors when the notebook is rerun.
if collection.count() == 0:
    for start in range(0, len(documents), batch_size):
        end = start + batch_size

        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]

        batch_embeddings = embedding_model.encode(
            batch_docs,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )

print("Chroma count:", collection.count())
