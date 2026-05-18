from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are an expert assistant analyzing California Department of Managed Health Care (DMHC) Independent Medical Review (IMR) determinations.

Your job is to answer the user's question using ONLY the IMR records provided. Each record describes a real insurance dispute, including the diagnosis, requested treatment, the health plan's original decision, and the independent reviewer's findings.

Rules:
1. Base your answer strictly on the provided records. Do not invent facts.
2. Cite each claim with the reference ID in square brackets, like [MN16-22851].
3. If the records do not contain enough information to answer the question, say so plainly.
4. Be concise. Prefer 2 to 4 short paragraphs over long prose.
5. Do not make medical or legal recommendations. You are summarizing publicly documented decisions, not advising the reader."""


USER_PROMPT_TEMPLATE = """## Records

{records}

## Question

{question}

## Answer"""


def format_record(index: int, result: dict[str, Any]) -> str:
    """Format one retrieved record for inclusion in the prompt."""
    p = result["payload"]
    findings = p.get("findings", "")
    # Truncate findings to keep prompts bounded; full text is huge for some records
    if len(findings) > 1500:
        findings = findings[:1500] + "..."
    return (
        f"[Record {index}]\n"
        f"Reference: {p.get('reference_id', 'unknown')}\n"
        f"Year: {p.get('report_year', 'unknown')}\n"
        f"Diagnosis: {p.get('diagnosis_category', 'unknown')} / "
        f"{p.get('diagnosis_subcategory', 'unknown')}\n"
        f"Treatment: {p.get('treatment_category', 'unknown')}\n"
        f"Determination: {p.get('determination', 'unknown')}\n"
        f"Findings: {findings}"
    )


def build_prompt(question: str, retrieved: list[dict[str, Any]]) -> tuple[str, str]:
    """Build the (system, user) prompts for the LLM.

    Args:
        question: The user's question.
        retrieved: List of search results from the vector store.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    if not retrieved:
        records_text = "(No relevant records were found in the dataset.)"
    else:
        records_text = "\n\n".join(
            format_record(i, r) for i, r in enumerate(retrieved, 1)
        )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        records=records_text, question=question
    )
    return SYSTEM_PROMPT, user_prompt
