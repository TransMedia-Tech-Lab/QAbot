"""esa.io API client and answer provider."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import requests

from .knowledge import clean_message

logger = logging.getLogger(__name__)

MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
MARKDOWN_EMPHASIS = re.compile(r"[*_`]")


class EsaClientError(RuntimeError):
    """Raised when the esa API cannot be reached or returns an error."""


@dataclass(frozen=True)
class EsaPost:
    """Simplified representation of an esa post."""

    title: str
    url: str
    body_md: str


class EsaClient:
    """Thin wrapper around the esa REST API."""

    def __init__(
        self,
        *,
        team: str,
        token: str,
        base_url: str = "https://api.esa.io/v1",
        session: Optional[requests.Session] = None,
        timeout: float = 5.0,
    ):
        self._team = team
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "User-Agent": "QAbot/esa-integration",
                "Content-Type": "application/json",
            }
        )

    def search_posts(self, query: str, *, per_page: int = 10) -> List[EsaPost]:
        if not query:
            return []

        url = f"{self._base_url}/teams/{self._team}/posts"
        params = {
            "q": query,
            "per_page": per_page,
            "sort": "updated",
            "order": "desc",
        }

        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise EsaClientError(f"esa API request failed: {exc}") from exc

        data = response.json()
        posts = []
        for item in data.get("posts", []):
            posts.append(
                EsaPost(
                    title=item.get("name", "Untitled"),
                    url=item.get("url") or "",
                    body_md=item.get("body_md") or item.get("body_html") or "",
                )
            )
        return posts

    def get_all_posts(self, *, per_page: int = 100, page: int = 1) -> List[EsaPost]:
        """
        全記事を取得（クエリなし）

        Args:
            per_page: 1ページあたりの記事数
            page: ページ番号

        Returns:
            記事のリスト
        """
        url = f"{self._base_url}/teams/{self._team}/posts"
        params = {
            "per_page": per_page,
            "page": page,
            "sort": "updated",
            "order": "desc",
        }

        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise EsaClientError(f"esa API request failed: {exc}") from exc

        data = response.json()
        posts = []
        for item in data.get("posts", []):
            posts.append(
                EsaPost(
                    title=item.get("name", "Untitled"),
                    url=item.get("url") or "",
                    body_md=item.get("body_md") or item.get("body_html") or "",
                )
            )
        return posts


class EsaAnswerProvider:
    """Answer generator that queries esa.io and caches recent results."""

    def __init__(self, client: EsaClient, cache_ttl_seconds: int = 600):
        self._client = client
        self._cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple[float, Optional[str]]] = {}

    def lookup(self, message_text: str) -> Optional[str]:
        normalized = clean_message(message_text)
        if not normalized:
            return None

        cache_key = normalized.lower()
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < self._cache_ttl:
            return cached[1]

        keywords = _extract_keywords(normalized)
        best_candidate: tuple[Optional[EsaPost], int] = (None, -1)
        for query in _build_queries(normalized, keywords):
            try:
                posts = self._client.search_posts(query)
            except EsaClientError:
                logger.exception("esa API 呼び出しに失敗しました (query=%s)", query)
                break
            if posts:
                selected, score = _select_best_post(posts, keywords)
                if selected and score > best_candidate[1]:
                    best_candidate = (selected, score)
                if keywords and score >= len(keywords):
                    break

        selected = best_candidate[0]
        answer: Optional[str] = None
        if selected:
            answer = format_post_answer(selected)

        self._cache[cache_key] = (now, answer)
        return answer


def format_post_answer(post: EsaPost, *, excerpt_length: int = 200) -> str:
    excerpt = _extract_excerpt(post.body_md)
    if not excerpt:
        excerpt = "関連するesa記事が見つかりました。詳細は以下のリンクをご確認ください。"

    excerpt = _truncate(excerpt, excerpt_length)
    link = post.url or "https://esa.io"
    return f"*{post.title}*\n{excerpt}\n<{link}|esaで続きを読む>"


def _extract_excerpt(markdown_body: str) -> str:
    for line in markdown_body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = stripped.lstrip("#*- ")
        text = MARKDOWN_LINK.sub(r"\1", stripped)
        text = MARKDOWN_EMPHASIS.sub("", text)
        text = re.sub(r"\s+", " ", text)
        if text:
            return text
    return ""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


STOPWORDS = {
    "これは",
    "それは",
    "どんな",
    "どの",
    "どれ",
    "この",
    "その",
    "あの",
    "について",
    "ですか",
    "です",
    "か",
    "を",
    "が",
    "は",
    "の",
    "教えて",
    "ください",
}


def _build_queries(text: str, keywords: Optional[List[str]] = None) -> Iterable[str]:
    """Generate progressively broader esa検索クエリ."""
    yielded: List[str] = []
    if text:
        yielded.append(text)
        yield text

    tokens = keywords if keywords is not None else _extract_keywords(text)

    if tokens:
        joined = " ".join(tokens)
        if joined not in yielded:
            yielded.append(joined)
            yield joined
        for token in tokens:
            if token not in yielded:
                yielded.append(token)
                yield token


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    current = ""
    current_class = None

    for ch in text:
        cls = _char_class(ch)
        if cls == "other":
            if current:
                tokens.append(current)
                current = ""
                current_class = None
            continue

        if current and cls != current_class:
            tokens.append(current)
            current = ch
        else:
            current += ch
        current_class = cls

    if current:
        tokens.append(current)

    return tokens


def _char_class(ch: str) -> str:
    if "一" <= ch <= "龠":
        return "kanji"
    if "ぁ" <= ch <= "ゖ":
        return "hiragana"
    if "ァ" <= ch <= "ヺ" or ch == "ー":
        return "katakana"
    if ch.isdigit():
        return "digit"
    lower = ch.lower()
    if "a" <= lower <= "z":
        return "latin"
    return "other"


def _extract_keywords(text: str) -> List[str]:
    keywords: List[str] = []
    for token in _tokenize(text):
        token = token.strip()
        if len(token) < 2 or token in STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords


def _select_best_post(posts: List[EsaPost], keywords: List[str]) -> tuple[Optional[EsaPost], int]:
    if not posts:
        return None, -1

    if not keywords:
        return posts[0], 0

    best_post = posts[0]
    best_score = -1
    lowered_keywords = [kw.lower() for kw in keywords]

    for post in posts:
        title = (post.title or "").lower()
        body = (post.body_md or "").lower()
        title_hits = sum(1 for kw in lowered_keywords if kw in title)
        body_hits = sum(1 for kw in lowered_keywords if kw in body)
        score = title_hits * 3 + body_hits
        if score > best_score:
            best_post = post
            best_score = score

    return best_post, best_score
