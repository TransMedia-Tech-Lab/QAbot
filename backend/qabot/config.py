"""Configuration helpers for the QAbot Slack application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


DEFAULT_RESPONSE = (
    "研究室に関する情報を見つけられませんでした。"
    "キーワードを変えてもう一度聞いてみてください。"
)
DEFAULT_ESA_BASE_URL = "https://api.esa.io/v1"
DEFAULT_GEMMA_MODEL_NAME = "google/gemma-3n-e2b-it"
DEFAULT_GEMMA_DEVICE = "cpu"
DEFAULT_ESA_MAX_ARTICLES = 3  # RAGで使用する記事の最大件数
DEFAULT_ESA_MAX_CHARS_PER_ARTICLE = 2000  # 1記事あたりの最大文字数
DEFAULT_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"  # エンベディングモデル
DEFAULT_VECTOR_SEARCH_TOP_K = 3  # ベクトル検索で取得する記事数
DEFAULT_VECTOR_INDEX_MAX_POSTS = 500  # インデックス化する記事の最大数
DEFAULT_USE_VECTOR_SEARCH = True  # ベクトル検索を使用するかどうか


@dataclass(slots=True)
class Settings:
    """Runtime configuration."""

    bot_token: str
    app_token: str
    signing_secret: Optional[str] = None
    esa_team: Optional[str] = None
    esa_api_token: Optional[str] = None
    esa_base_url: str = DEFAULT_ESA_BASE_URL
    default_response: str = DEFAULT_RESPONSE
    gemma_model_name: str = DEFAULT_GEMMA_MODEL_NAME
    gemma_device: str = DEFAULT_GEMMA_DEVICE
    esa_max_articles: int = DEFAULT_ESA_MAX_ARTICLES
    esa_max_chars_per_article: int = DEFAULT_ESA_MAX_CHARS_PER_ARTICLE
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    vector_search_top_k: int = DEFAULT_VECTOR_SEARCH_TOP_K
    vector_index_max_posts: int = DEFAULT_VECTOR_INDEX_MAX_POSTS
    use_vector_search: bool = DEFAULT_USE_VECTOR_SEARCH

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        app_token = os.getenv("SLACK_APP_TOKEN")
        signing_secret = os.getenv("SLACK_SIGNING_SECRET")
        default_response = os.getenv("LAB_DEFAULT_RESPONSE", DEFAULT_RESPONSE)
        esa_team = os.getenv("ESA_TEAM")
        esa_api_token = os.getenv("ESA_API_TOKEN")
        esa_base_url = os.getenv("ESA_BASE_URL", DEFAULT_ESA_BASE_URL)
        gemma_model_name = os.getenv("GEMMA_MODEL_NAME", DEFAULT_GEMMA_MODEL_NAME)
        gemma_device = os.getenv("GEMMA_DEVICE", DEFAULT_GEMMA_DEVICE)
        esa_max_articles = int(os.getenv("ESA_MAX_ARTICLES", str(DEFAULT_ESA_MAX_ARTICLES)))
        esa_max_chars_per_article = int(
            os.getenv("ESA_MAX_CHARS_PER_ARTICLE", str(DEFAULT_ESA_MAX_CHARS_PER_ARTICLE))
        )
        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME)
        vector_search_top_k = int(os.getenv("VECTOR_SEARCH_TOP_K", str(DEFAULT_VECTOR_SEARCH_TOP_K)))
        vector_index_max_posts = int(
            os.getenv("VECTOR_INDEX_MAX_POSTS", str(DEFAULT_VECTOR_INDEX_MAX_POSTS))
        )
        use_vector_search = os.getenv("USE_VECTOR_SEARCH", str(DEFAULT_USE_VECTOR_SEARCH)).lower() in ("true", "1", "yes")

        missing = [
            name
            for name, value in (("SLACK_BOT_TOKEN", bot_token), ("SLACK_APP_TOKEN", app_token))
            if not value
        ]
        if missing:
            raise ConfigError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return cls(
            bot_token=bot_token,
            app_token=app_token,
            signing_secret=signing_secret,
            esa_team=esa_team,
            esa_api_token=esa_api_token,
            esa_base_url=esa_base_url if esa_base_url else DEFAULT_ESA_BASE_URL,
            default_response=default_response if default_response else DEFAULT_RESPONSE,
            gemma_model_name=gemma_model_name if gemma_model_name else DEFAULT_GEMMA_MODEL_NAME,
            gemma_device=gemma_device if gemma_device else DEFAULT_GEMMA_DEVICE,
            esa_max_articles=esa_max_articles,
            esa_max_chars_per_article=esa_max_chars_per_article,
            embedding_model_name=embedding_model_name if embedding_model_name else DEFAULT_EMBEDDING_MODEL_NAME,
            vector_search_top_k=vector_search_top_k,
            vector_index_max_posts=vector_index_max_posts,
            use_vector_search=use_vector_search,
        )
