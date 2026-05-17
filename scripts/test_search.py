import argparse
import logging

from app.rag.embedder import embed_text
from app.rag.vectorstore import search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the IMR index.")
    parser.add_argument("query", help="Natural-language search query.")
    parser.add_argument("--top-k", type=int, default=5, help="How many results to show.")
    parser.add_argument(
        "--filter-diagnosis",
        default=None,
        help="Filter results by diagnosis_category (e.g. 'Mental Disorder').",
    )
    args = parser.parse_args()

    filters = {}
    if args.filter_diagnosis:
        filters["diagnosis_category"] = args.filter_diagnosis

    query_vector = embed_text(args.query)
    results = search(query_vector, top_k=args.top_k, filters=filters or None)

    print(f"\nQuery: {args.query}")
    if filters:
        print(f"Filters: {filters}")
    print(f"Top {len(results)} results:\n")

    for i, r in enumerate(results, 1):
        p = r["payload"]
        print(f"--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"Reference: {p.get('reference_id')}")
        print(f"Diagnosis: {p.get('diagnosis_category')} / {p.get('diagnosis_subcategory')}")
        print(f"Determination: {p.get('determination')}")
        print(f"Findings: {p.get('findings', '')[:200]}...\n")


if __name__ == "__main__":
    main()
