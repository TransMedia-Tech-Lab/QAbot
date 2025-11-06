"""Configuration helpers for the QAbot Slack application."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(slots=True)
class Settings:
    """Runtime configuration."""

    bot_token: str
    app_token: str
    signing_secret: Optional[str] = None
    default_response: str = (
        "研究室に関する情報を見つけられませんでした。"
        "キーワードを変えてもう一度聞いてみてください。"
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        app_token = os.getenv("SLACK_APP_TOKEN")
        signing_secret = os.getenv("SLACK_SIGNING_SECRET")
        default_response = os.getenv("LAB_DEFAULT_RESPONSE")

        missing = [name for name, value in (("SLACK_BOT_TOKEN", bot_token), ("SLACK_APP_TOKEN", app_token)) if not value]
        if missing:
            raise ConfigError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return cls(
            bot_token=bot_token,
            app_token=app_token,
            signing_secret=signing_secret,
            default_response=default_response
            if default_response
            else Settings.default_response,
        )

