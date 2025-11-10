"""ResearchLabBot Slack実装."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv
from loguru import logger
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from .config import (
    DEFAULT_CHROMA_PERSIST_DIRECTORY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_FILE_BOT,
)
from .esa_client import EsaClient
from .llm_manager import LLMManager
from .vector_store import RAGEngine, VectorStore


class ResearchLabBot:
    """esa記事 + RAG で回答するSlackボット."""

    def __init__(self) -> None:
        load_dotenv()
        self._setup_logging()
        self._initialize_components()
        self.app = App(
            token=os.environ.get("SLACK_BOT_TOKEN"),
            signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
        )
        self._setup_event_handlers()
        logger.info("ResearchLabBot initialised")

    def _setup_logging(self) -> None:
        log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
        log_file = os.getenv("LOG_FILE", DEFAULT_LOG_FILE_BOT)
        log_dir = os.path.dirname(log_file) or "."
        os.makedirs(log_dir, exist_ok=True)
        logger.add(
            log_file,
            rotation="1 day",
            retention="7 days",
            level=log_level,
        )

    def _initialize_components(self) -> None:
        access_token = os.environ.get("ESA_ACCESS_TOKEN") or os.environ.get("ESA_API_TOKEN")
        team_name = os.environ.get("ESA_TEAM_NAME") or os.environ.get("ESA_TEAM")
        if not access_token or not team_name:
            raise RuntimeError("ESA_ACCESS_TOKEN/ESA_TEAM_NAME もしくは ESA_API_TOKEN/ESA_TEAM を設定してください。")

        self.esa_client = EsaClient(access_token=access_token, team_name=team_name)
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_CHROMA_PERSIST_DIRECTORY)
        os.makedirs(persist_directory, exist_ok=True)
        self.vector_store = VectorStore(
            persist_directory=persist_directory,
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        )
        self.rag_engine = RAGEngine(self.vector_store)
        self.llm_manager = LLMManager()

    def _setup_event_handlers(self) -> None:
        @self.app.event("app_mention")
        def handle_mention(event, say):
            self._handle_mention(event, say)

        @self.app.event("message")
        def handle_message(event, say):
            if event.get("channel_type") == "im":
                self._handle_message(event, say)

        @self.app.command("/lab")
        def handle_lab_command(ack, command):
            ack()
            self._handle_command(command)

    def _handle_mention(self, event: Dict, say) -> None:
        try:
            user = event.get("user")
            text = event.get("text", "")
            channel = event.get("channel")
            question = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

            if not question:
                say("質問を入力してください。例: `@bot 研究室の鍵番号は？`")
                return

            logger.info("質問受信: %s (from %s)", question, user)
            thinking_msg = say("🔍 情報を検索中...")
            answer, urls = self._generate_answer(question)
            response = self._format_response(question, answer, urls)

            self.app.client.chat_update(
                channel=channel,
                ts=thinking_msg["ts"],
                text=response,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("メンション処理エラー: %s", exc)
            say("申し訳ありません。エラーが発生しました。")

    def _handle_message(self, event: Dict, say) -> None:
        if event.get("bot_id"):
            return
        text = event.get("text", "")
        user = event.get("user")
        logger.info("DM受信: %s (from %s)", text, user)
        answer, urls = self._generate_answer(text)
        say(self._format_response(text, answer, urls))

    def _handle_command(self, command: Dict) -> None:
        text = command.get("text", "").strip()
        response_url = command.get("response_url")

        if text == "sync":
            self._sync_database()
            self._send_response(response_url, "✅ データベースの同期が完了しました。")
        elif text == "stats":
            self._send_response(response_url, self._get_stats())
        elif text.startswith("search "):
            query = text[7:]
            answer, urls = self._generate_answer(query)
            self._send_response(response_url, self._format_response(query, answer, urls))
        else:
            help_text = (
                "*利用可能なコマンド:*\n"
                "• `/lab search [質問]` - 情報を検索\n"
                "• `/lab sync` - データベースを同期（管理者のみ）\n"
                "• `/lab stats` - 統計情報を表示\n"
                "• `/lab help` - このヘルプを表示"
            )
            self._send_response(response_url, help_text)

    def _generate_answer(self, question: str) -> tuple[str, list[str]]:
        try:
            search_results = self.rag_engine.search_and_rank(question, top_k=5)
            if not search_results:
                return "申し訳ありません。関連する情報が見つかりませんでした。", []
            context = self.rag_engine.format_context(search_results)
            answer = self.llm_manager.generate_answer(question, context)
            urls = self.rag_engine.get_source_urls(search_results)
            return answer, urls
        except Exception as exc:  # noqa: BLE001
            logger.exception("回答生成エラー: %s", exc)
            return "申し訳ありません。回答の生成中にエラーが発生しました。", []

    def _format_response(self, question: str, answer: str, urls: list[str]) -> str:
        response_parts = [f"> {question}", "", answer]
        if urls:
            response_parts.append("")
            response_parts.append("📚 *参照記事:*")
            for idx, url in enumerate(urls[:3], 1):
                response_parts.append(f"{idx}. <{url}|記事を見る>")
        return "\n".join(response_parts)

    def _sync_database(self) -> None:
        logger.info("データベース同期開始")
        posts = self.esa_client.get_all_posts()
        logger.info("取得した記事数: %d", len(posts))
        self.vector_store.add_documents(posts)
        logger.info("データベース同期完了")

    def _get_stats(self) -> str:
        try:
            collection_stats = self.vector_store.collection.count()
            stats = (
                "📊 *ボット統計情報:*\n"
                f"• インデックス済みチャンク数: {collection_stats}\n"
                f"• 埋め込みモデル: {self.vector_store.embedding_model_name}\n"
                f"• LLMプロバイダ: {type(self.llm_manager.provider).__name__}\n"
                f"• 最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return stats
        except Exception as exc:  # noqa: BLE001
            logger.exception("統計情報取得エラー: %s", exc)
            return "統計情報の取得に失敗しました。"

    def _send_response(self, response_url: str, text: str) -> None:
        import requests

        try:
            requests.post(response_url, json={"text": text}, timeout=10)
        except Exception as exc:  # noqa: BLE001
            logger.error("レスポンス送信エラー: %s", exc)

    def run(self) -> None:
        try:
            handler = SocketModeHandler(self.app, os.environ["SLACK_APP_TOKEN"])
            logger.info("ResearchLabBot起動中...")
            handler.start()
        except KeyboardInterrupt:
            logger.info("ボット停止要求を受信しました")
        except Exception as exc:  # noqa: BLE001
            logger.exception("起動エラー: %s", exc)
            raise
