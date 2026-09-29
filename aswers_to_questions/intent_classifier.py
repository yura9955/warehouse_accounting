from . import config


def classify_intent(question: str) -> tuple[str, float, dict[str, int]]:
    """Определяет интент вопроса правилами.

    Args:
        question: вопрос пользователя.

    Returns:
        (intent, confidence, scores):
            intent     — один из config.INTENT_KEYWORDS или "unknown";
            confidence — (top1 - top2) / top1, диапазон [0.0, 1.0];
            scores     — счётчики по интентам (для отладки и RESULTS.md).
    """
    text = question.lower().strip()
    if not text:
        return "unknown", 0.0, {}

    if any(phrase in text for phrase in config.UNKNOWN_PHRASES):
        return "unknown", 0.0, {}

    scores: dict[str, int] = {}
    for intent, keywords in config.INTENT_KEYWORDS.items():
        score = _score(text, keywords)
        if score > 0:
            scores[intent] = score

    if not scores:
        return "unknown", 0.0, {}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top1_intent, top1 = ranked[0]
    top2 = ranked[1][1] if len(ranked) > 1 else 0

    confidence = (top1 - top2) / top1
    return top1_intent, float(confidence), scores


def _score(text: str, keywords: list[str]) -> int:
    """Суммирует длины совпавших фраз, не учитывая вложенные дважды."""
    matched: list[str] = []
    for kw in sorted(keywords, key=len, reverse=True):
        if kw in text and not any(kw in m for m in matched):
            matched.append(kw)
    return sum(len(kw) for kw in matched)