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
        )
