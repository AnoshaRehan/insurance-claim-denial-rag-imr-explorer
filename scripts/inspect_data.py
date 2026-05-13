"""Quick inspection of the loaded IMR data."""
import logging
from collections import Counter

from app.data.loader import load_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    print("Loading records (will fetch from source on first run)...")
    records = load_records()
    print(f"\nLoaded {len(records):,} records\n")

    # Sample one
    sample = records[0]
    print("Example record:")
    print(f"  Reference ID:  {sample.reference_id}")
    print(f"  Year:          {sample.report_year}")
    print(f"  Diagnosis:     {sample.diagnosis_category} / {sample.diagnosis_subcategory}")
    print(f"  Treatment:     {sample.treatment_category}")
    print(f"  Determination: {sample.determination}")
    print(f"  Type:          {sample.type}")
    print(f"  Findings:      {sample.findings[:200]}...\n")

    # Distribution of determinations
    print("Determinations:")
    for det, count in Counter(r.determination for r in records).most_common():
        print(f"  {det}: {count:,}")

    # Top diagnosis categories
    print("\nTop 10 diagnosis categories:")
    for cat, count in Counter(r.diagnosis_category for r in records).most_common(10):
        print(f"  {cat}: {count:,}")


if __name__ == "__main__":
    main()
