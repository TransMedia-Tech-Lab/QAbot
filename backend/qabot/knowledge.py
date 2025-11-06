"""Keyword-based knowledge base for lab-related questions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional


MENTION_PATTERN = re.compile(r"<@[^>]+>")


@dataclass(frozen=True)
class KnowledgeEntry:
    keywords: tuple[str, ...]
    answer: str

    def matches(self, text: str) -> bool:
        normalized = text.lower()
        return any(keyword.lower() in normalized for keyword in self.keywords)


def clean_message(text: str) -> str:
    """Remove Slack mention tags and trim whitespace."""
    without_mentions = MENTION_PATTERN.sub(" ", text or "")
    return " ".join(without_mentions.split())


def build_default_entries() -> List[KnowledgeEntry]:
    return [
        KnowledgeEntry(
            keywords=("研究テーマ", "テーマ", "focus", "research"),
            answer=(
                "当研究室では人と協調するAIシステムをテーマに、"
                "自然言語処理とロボティクスを横断した応用研究を進めています。"
                "年間を通して産学連携プロジェクトにも参加しています。"
            ),
        ),
        KnowledgeEntry(
            keywords=("メンバー", "人数", "構成", "student"),
            answer=(
                "現在は教授1名、助教1名、博士課程3名、修士8名、学部4名が所属しています。"
                "留学生も受け入れており、英語での議論も日常的に行われます。"
            ),
        ),
        KnowledgeEntry(
            keywords=("スケジュール", "ミーティング", "ゼミ", "meeting"),
            answer=(
                "毎週火曜午後に進捗報告ゼミ、金曜午後に論文輪講を実施しています。"
                "その他、必要に応じてプロジェクトごとの小ミーティングを設定しています。"
            ),
        ),
        KnowledgeEntry(
            keywords=("設備", "環境", "gpu", "サーバ", "machine"),
            answer=(
                "研究室にはGPUサーバ4台とロボット実験スペースがあり、"
                "実験予約はSlackの#lab-facilityチャンネルで管理しています。"
            ),
        ),
        KnowledgeEntry(
            keywords=("募集", "見学", "join", "応募", "インターン"),
            answer=(
                "見学希望はSlackでメンションするか、lab-admin@example.com までご連絡ください。"
                "毎月第3金曜にオンライン説明会も開催しています。"
            ),
        ),
    ]


DEFAULT_ENTRIES = build_default_entries()


def lookup_answer(message_text: str, entries: Optional[Iterable[KnowledgeEntry]] = None) -> Optional[str]:
    """Return the best answer for the incoming text."""
    cleaned = clean_message(message_text)
    for entry in entries or DEFAULT_ENTRIES:
        if entry.matches(cleaned):
            return entry.answer
    return None
