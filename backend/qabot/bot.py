"""Slack Bolt application wiring."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from . import knowledge
from .config import Settings
from .esa import EsaAnswerProvider, EsaClient
from .gemma_provider import GemmaAnswerProvider


class LabSlackBot:
    """Registers Slack handlers and starts the Socket Mode loop."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._logger = logging.getLogger(__name__)
        self._esa_provider: Optional[EsaAnswerProvider] = self._build_esa_provider()
        self._gemma_provider: Optional[GemmaAnswerProvider] = self._build_gemma_provider()
        self._app = App(
            token=settings.bot_token,
            signing_secret=settings.signing_secret,
        )
        self._register_handlers()

    def _build_esa_provider(self) -> Optional[EsaAnswerProvider]:
        if not (self._settings.esa_team and self._settings.esa_api_token):
            self._logger.info("esa連携は未設定です（ESA_TEAM / ESA_API_TOKEN が見つかりません）")
            return None

        client = EsaClient(
            team=self._settings.esa_team,
            token=self._settings.esa_api_token,
            base_url=self._settings.esa_base_url,
        )
        self._logger.info("esa連携を有効化しました（team=%s）", self._settings.esa_team)
        return EsaAnswerProvider(client)

    def _build_gemma_provider(self) -> Optional[GemmaAnswerProvider]:
        """Gemma Provider を初期化（常に有効化）"""
        try:
            self._logger.info("Gemma Provider を初期化中...")
            provider = GemmaAnswerProvider(
                model_name=self._settings.gemma_model_name,
                device=self._settings.gemma_device
            )
            self._logger.info("Gemma Provider を有効化しました")
            return provider
        except Exception as e:
            self._logger.error(f"Gemma Provider の初期化に失敗しました: {e}", exc_info=True)
            return None

    def _register_handlers(self) -> None:
        @self._app.event("app_mention")
        def handle_app_mention(body: Dict[str, Any], say, logger) -> None:  # type: ignore[no-untyped-def]
            event = body.get("event", {})
            text = event.get("text", "")
            thread_ts = event.get("thread_ts") or event.get("ts")
            channel = event.get("channel")
            logger.info("app_mention received in %s: %s", channel, text)

            if not channel:
                logger.error("Cannot reply: channel not found in event payload")
                return

            # スレッドIDを生成（チャンネル + スレッドタイムスタンプ）
            thread_id = f"{channel}:{thread_ts}" if thread_ts else f"{channel}:default"
            response = self._build_response(text, thread_id)

            say_kwargs: Dict[str, Any] = {"text": response, "channel": channel}
            if thread_ts:
                say_kwargs["thread_ts"] = thread_ts

            say(**say_kwargs)

        @self._app.event("message")
        def handle_direct_message(body: Dict[str, Any], say, logger) -> None:  # type: ignore[no-untyped-def]
            event = body.get("event", {})
            if event.get("channel_type") != "im" or event.get("bot_id") or event.get("subtype"):
                return

            text = event.get("text", "")
            channel = event.get("channel")
            user = event.get("user")
            logger.info("DM received from %s: %s", user, text)

            if not channel:
                logger.error("Cannot reply to DM: channel missing in payload")
                return

            # DMはユーザーごとに会話履歴を管理
            thread_id = f"dm:{user}"
            response = self._build_response(text, thread_id)
            say(text=response, channel=channel)

        @self._app.event("app_home_opened")
        def handle_app_home_opened(body: Dict[str, Any], client, logger) -> None:  # type: ignore[no-untyped-def]
            user_id = body["event"]["user"]
            logger.debug("App home opened by %s", user_id)
            client.views_publish(
                user_id=user_id,
                view={
                    "type": "home",
                    "blocks": [
                        {"type": "section", "text": {"type": "mrkdwn", "text": "*QAbotへようこそ*"}},
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "Gemmaモデルを使用した研究室向けチャットボットです。メンションまたはDMで質問してください。",
                            },
                        },
                        {"type": "divider"},
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "💡 *使い方*\n• チャンネルでメンション: `@QAbot 質問内容`\n• DM: 直接メッセージを送信\n• スレッド: スレッド内で会話履歴を保持",
                            },
                        },
                    ],
                },
            )

    def start(self) -> None:
        """Start Socket Mode handler."""
        handler = SocketModeHandler(self._app, self._settings.app_token)
        handler.start()

    def _build_response(self, message_text: str, thread_id: str) -> str:
        """メッセージに対する応答を生成"""
        # メンション記号を削除してクリーンなメッセージを取得
        cleaned_message = knowledge.clean_message(message_text)

        # Gemma Providerを使用して応答を生成
        if self._gemma_provider:
            try:
                return self._gemma_provider.get_response(thread_id, cleaned_message)
            except Exception as e:
                self._logger.error(f"Gemma応答生成に失敗: {e}", exc_info=True)

        # Gemmaが使えない場合は従来のキーワードベース応答にフォールバック
        answer = None
        if self._esa_provider:
            answer = self._esa_provider.lookup(cleaned_message)
        if not answer:
            answer = knowledge.lookup_answer(cleaned_message)
        return answer if answer else self._settings.default_response
