import re


SQL_PATTERNS = [
    r"\bdatasheets?\b",
    r"\btoughness\b",
    r"\bweapons?\b",
    r"\bwargear\b",
    r"\bkeywords?\b",
    r"\babilities\b",
    r"\bcharacteristics?\b",
    r"\bstats?\b",
    r"\bprofiles?\b",
    r"\bfactions?\b",
    r"\bdetachments?\b",
    r"\bstratagems?\b",
    r"\benhancements?\b",
    r"\bstrength\b",
    r"\bwounds\b",
    r"\bdamage\b",
    r"\bpoints?\s+cost\b",
    r"\bleaders?\b",
    r"\b(?:lead|leads|leading)\s+(?:a\s+)?(?:unit|units|terminators?)\b",
]

PDF_PATTERNS = [
    r"\bcore rules?\b",
    r"\bwound rolls?\b",
    r"\bhit rolls?\b",
    r"\bsaving throws?\b",
    r"\bengagement range\b",
    r"\bbattle rounds?\b",
    r"\bphases?\b",
    r"\badvanc(?:e|es|ed|ing)\b",
    r"\bfall(?:ing)? back\b",
    r"\bshoot(?:ing)?\b",
    r"\bcharg(?:e|es|ed|ing)\b",
    r"\bfight(?:ing)?\b",
    r"\bto wound\b",
    r"\bto hit\b",
    r"\b(?:how|when)\s+.*\b(?:work|resolve|determine)\b",
]

NON_ENTITY_WORDS = {
    "what", "how", "when", "where", "why", "which",
    "who", "can", "does", "do", "is", "are",
    "the", "a", "an", "if",
    "core", "rules", "fight", "shooting",
    "movement", "charge", "command", "phase",
    "advance", "advancing", "engagement", "range",
    "strength", "toughness", "weapon", "weapons",
    "unit", "units", "wound", "roll", "terminator",
}


def _matches(question: str, patterns: list[str]) -> bool:
    """Check whether a question matches any routing pattern."""
    return any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in patterns
    )


def _mentions_named_entity(question: str) -> bool:
    """Detect possible named units using capitalization heuristics."""
    tokens = re.findall(r"\b[A-Z][A-Za-z]*(?:['’]s)?\b", question)

    return any(
        token.lower().removesuffix("'s").removesuffix("’s")
        not in NON_ENTITY_WORDS
        for token in tokens
    )


def route_question(question: str) -> str:
    """Return 'pdf', 'sql', or 'mixed'.

    Ambiguous queries default to mixed retrieval rather than
    risk excluding a potentially necessary information source.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    sql_signal = _matches(question, SQL_PATTERNS)
    pdf_signal = _matches(question, PDF_PATTERNS)
    named_entity = _mentions_named_entity(question)

    if sql_signal and pdf_signal:
        return "mixed"

    if pdf_signal and named_entity:
        return "mixed"

    if sql_signal:
        return "sql"

    if pdf_signal:
        return "pdf"

    return "mixed"
