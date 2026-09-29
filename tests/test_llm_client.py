"""Тесты разбора JSON-ответа LLM (без реального вызова API)."""
from __future__ import annotations

import pytest

from aswers_to_questions.llm_client import _parse_json


class TestParseJson:
    """Извлечение JSON из ответа модели, включая markdown-обёртку."""

    def test_plain_json(self):
        assert _parse_json('{"intent": "budget"}') == {"intent": "budget"}

    def test_markdown_fence_with_lang(self):
        raw = '```json\n{"intent": "budget"}\n```'
        assert _parse_json(raw) == {"intent": "budget"}

    def test_markdown_fence_without_lang(self):
        raw = '```\n{"intent": "budget"}\n```'
        assert _parse_json(raw) == {"intent": "budget"}

    def test_empty_returns_none(self):
        assert _parse_json("") is None

    def test_invalid_returns_none(self):
        assert _parse_json("не json") is None

    def test_list_is_not_dict(self):
        # Верхний уровень должен быть объектом, а не массивом
        assert _parse_json('[1, 2, 3]') is None