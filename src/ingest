import json
import pandas as pd
from pathlib import Path


def safe_get(data, path, default=None):
    current = data

    for key in path:
        if isinstance(current, dict):
            current = current.get(key, default)
        elif isinstance(current, list) and isinstance(key, int):
            current = current[key] if len(current) > key else default
        else:
            return default

        if current is None:
            return default

    return current


def extract_suppliers(record):
    suppliers = []

    for award in record.get("awards", []):
        for supplier in award.get("suppliers", []):
            name = supplier.get("name")
            if name:
                suppliers.append(name)

    return sorted(set(suppliers))


def extract_cpv(record):
    cpv_codes = []
    cpv_descriptions = []

    classification = safe_get(record, ["tender", "classification"], {})
    if isinstance(classification, dict):
        if classification.get("id"):
            cpv_codes.append(classification["id"])
        if classification.get("description"):
            cpv_descriptions.append(classification["description"])

    for item in safe_get(record, ["tender", "items"], []):
        for cpv in item.get("additionalClassifications", []):
            if cpv.get("id"):
                cpv_codes.append(cpv["id"])
            if cpv.get("description"):
                cpv_descriptions.append(cpv["description"])

    return sorted(set(cpv_codes)), sorted(set(cpv_descriptions))


def extract_regions(record):
    regions = []

    for item in safe_get(record, ["tender", "items"], []):
        for address in item.get("deliveryAddresses", []):
            if address.get("region"):
                regions.append(address["region"])

        delivery_location = item.get("deliveryLocation", {})
        if isinstance(delivery_location, dict) and delivery_location.get("description"):
            regions.append(delivery_location["description"])

    return sorted(set(regions))


def extract_notice_url(record):
    documents = safe_get(record, ["tender", "documents"], [])

    for doc in documents:
        if doc.get("url"):
            return doc["url"]

    planning_docs = safe_get(record, ["planning", "documents"], [])

    for doc in planning_docs:
        if doc.get("url"):
            return doc["url"]

    return None


def extract_contract_values(record):
    values = []

    tender_amount = safe_get(record, ["tender", "value", "amount"])
    if tender_amount is not None:
        values.append(tender_amount)

    for contract in record.get("contracts", []):
        amount = safe_get(contract, ["value", "amount"])
        if amount is not None:
            values.append(amount)

    return values


def extract_award_statuses(record):
    return sorted(set([
        award.get("status")
        for award in record.get("awards", [])
        if award.get("status")
    ]))


def extract_contract_statuses(record):
    return sorted(set([
        contract.get("status")
        for contract in record.get("contracts", [])
        if contract.get("status")
    ]))


def flatten_ocds_record(record):
    cpv_codes, cpv_descriptions = extract_cpv(record)
    suppliers = extract_suppliers(record)
    regions = extract_regions(record)
    contract_values = extract_contract_values(record)

    title = safe_get(record, ["tender", "title"])
    description = safe_get(record, ["tender", "description"])
    buyer_name = safe_get(record, ["buyer", "name"])

    search_text = " ".join([
        str(title or ""),
        str(description or ""),
        str(buyer_name or ""),
        " ".join(suppliers),
        " ".join(cpv_descriptions),
        " ".join(regions),
        str(safe_get(record, ["tender", "procurementMethod"]) or ""),
        str(safe_get(record, ["tender", "mainProcurementCategory"]) or "")
    ])

    return {
        "ocid": record.get("ocid"),
        "notice_id": record.get("id"),
        "date": record.get("date"),

        "title": title,
        "description": description,

        "buyer_id": safe_get(record, ["buyer", "id"]),
        "buyer_name": buyer_name,

        "supplier_names": ", ".join(suppliers),

        "cpv_codes": ", ".join(cpv_codes),
        "cpv_descriptions": ", ".join(cpv_descriptions),

        "regions": ", ".join(regions),

        "tender_status": safe_get(record, ["tender", "status"]),
        "award_statuses": ", ".join(extract_award_statuses(record)),
        "contract_statuses": ", ".join(extract_contract_statuses(record)),

        "procurement_method": safe_get(record, ["tender", "procurementMethod"]),
        "procurement_method_details": safe_get(record, ["tender", "procurementMethodDetails"]),
        "main_procurement_category": safe_get(record, ["tender", "mainProcurementCategory"]),

        "contract_values": contract_values,
        "max_contract_value": max(contract_values) if contract_values else None,

        "notice_url": extract_notice_url(record),

        "search_text": search_text
    }


def read_jsonl_and_flatten(file_path):
    flattened_records = []
    failed_lines = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                flattened = flatten_ocds_record(record)
                flattened_records.append(flattened)

            except Exception as e:
                failed_lines.append({
                    "line_number": line_number,
                    "error": str(e)
                })

    df = pd.DataFrame(flattened_records)
    failed_df = pd.DataFrame(failed_lines)

    return df, failed_df

file_path = "united_kingdom_fts_2026.jsonl"

df_contracts, failed_df = read_jsonl_and_flatten(file_path)

print("Total contracts processed:", len(df_contracts))
print("Failed lines:", len(failed_df))

df_contracts.head()
