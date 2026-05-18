from __future__ import annotations

import argparse
import json
import logging

from app.rag.pipeline import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question to the IMR RAG system.")
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of records to retrieve (default: from config).",
    )
    parser.add_argument(
        "--filter-diagnosis",
        default=None,
        help="Filter sources by diagnosis category (e.g. 'Mental Disorder').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable format.",
    )
    args = parser.parse_args()

    filters = {}
    if args.filter_diagnosis:
        filters["diagnosis_category"] = args.filter_diagnosis

    result = answer_question(
        question=args.question,
        top_k=args.top_k,
        filters=filters or None,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return
    
    print(f"\n{'=' * 70}")
    print(f"Q: {result['question']}")
    print(f"{'=' * 70}\n")
    print(result["answer"])

    if result["low_confidence"]:
        print(
            f"\nNo records in the dataset met the relevance threshold. "
            f"The question may be outside this dataset's scope."
        )
    else:
        print(f"\n{'-' * 70}")
        print("Sources used:")
        for i, src in enumerate(result["sources"], 1):
            print(
                f"{i}. [{src['reference_id']}] "
                f"{src['diagnosis_category']} | "
                f"{src['determination']} "
                f"(score: {src['score']:.3f})"
            )


if __name__ == "__main__":
    main()
