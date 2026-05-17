from __future__ import annotations

import argparse
import logging

from app.data.loader import IMRRecord, load_records
from app.rag.embedder import embed_texts
from app.rag.vectorstore import count_points, ensure_collection, upsert_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def record_to_text(record: IMRRecord) -> str:
    """Combine the record's structured fields and findings into one text blob."""
    return (
        f"Diagnosis: {record.diagnosis_category} / {record.diagnosis_subcategory}\n"
        f"Treatment: {record.treatment_category} / {record.treatment_subcategory}\n"
        f"Type: {record.type}\n"
        f"Determination: {record.determination}\n"
        f"Findings: {record.findings}"
    )


def record_to_payload(record: IMRRecord) -> dict:
    """Build the metadata payload stored alongside the vector in Qdrant."""
    return {
        "reference_id": record.reference_id,
        "report_year": record.report_year,
        "diagnosis_category": record.diagnosis_category,
        "diagnosis_subcategory": record.diagnosis_subcategory,
        "treatment_category": record.treatment_category,
        "treatment_subcategory": record.treatment_subcategory,
        "determination": record.determination,
        "type": record.type,
        "age_range": record.age_range,
        "patient_gender": record.patient_gender,
        "findings": record.findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Qdrant index.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only embed the first N records (useful for quick testing).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection (wipes existing index).",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force re-download of the source dataset.",
    )
    args = parser.parse_args()

    # Step 1: Load records
    logger.info("Loading records...")
    records = load_records(force_refresh=args.refresh_data)
    if args.limit:
        records = records[: args.limit]
        logger.info("Limited to first %d records for testing.", args.limit)
    logger.info("Loaded %d records", len(records))

    # Step 2: Prepare collection
    ensure_collection(recreate=args.recreate)

    # Step 3: Generate embeddings
    logger.info("Generating embeddings (this may take a few minutes)...")
    texts = [record_to_text(r) for r in records]
    vectors = embed_texts(texts)
    logger.info("Generated %d embeddings", len(vectors))

    # Step 4: Upload to Qdrant
    logger.info("Uploading to Qdrant...")
    payloads = [record_to_payload(r) for r in records]
    upsert_points(vectors, payloads)

    # Step 5: Verify
    total = count_points()
    logger.info("Done. Collection now contains %d points.", total)


if __name__ == "__main__":
    main()
