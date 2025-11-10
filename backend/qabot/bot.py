"""Slack Bolt application wiring."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from . import knowledge
from .config import Settings
from .esa import EsaAnswerProvider, EsaClient, EsaClientError, EsaPost
from .gemma_provider import GemmaAnswerProvider
from .vector_store import EsaVectorStore


class LabSlackBot:
    """Registers Slack handlers and starts the Socket Mode loop."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._logger = logging.getLogger(__name__)
        self._esa_client: Optional[EsaClient] = self._build_esa_client()
        self._esa_provider: Optional[EsaAnswerProvider] = self._build_esa_provider()
        self._vector_store: Optional[EsaVectorStore] = self._build_vector_store()
        self._gemma_provider: Optional[GemmaAnswerProvider] = self._build_gemma_provider()
        self._app = App(
            token=settings.bot_token,
            signing_secret=settings.signing_secret,
        )
        self._register_handlers()

    def _build_esa_client(self) -> Optional[EsaClient]:
        """esa.io APIクライアントを初期化"""
        if not (self._settings.esa_team and self._settings.esa_api_token):
            self._logger.info("esa連携は未設定です（ESA_TEAM / ESA_API_TOKEN が見つかりません）")
            return None

        client = EsaClient(
            team=self._settings.esa_team,
            token=self._settings.esa_api_token,
            base_url=self._settings.esa_base_url,
        )
        self._logger.info("esa連携を有効化しました（team=%s）", self._settings.esa_team)
        return client

    def _build_esa_provider(self) -> Optional[EsaAnswerProvider]:
        """esa Answer Providerを初期化（キーワードベース検索用）"""
        if not self._esa_client:
            return None
        return EsaAnswerProvider(self._esa_client)

    def _build_vector_store(self) -> Optional[EsaVectorStore]:
        """ベクトルストアを初期化してインデックスを構築"""
        if not self._esa_client or not self._settings.use_vector_search:
            if not self._settings.use_vector_search:
                self._logger.info("ベクトル検索は無効化されています（USE_VECTOR_SEARCH=False）")
            return None

        try:
            self._logger.info("ベクトルストアを初期化中...")
            vector_store = EsaVectorStore(
                esa_client=self._esa_client,
                embedding_model_name=self._settings.embedding_model_name,
                device=self._settings.gemma_device  # Gemmaと同じデバイスを使用
            )

            # インデックスを構築
            indexed_count = vector_store.build_index(
                max_posts=self._settings.vector_index_max_posts
            )

            if indexed_count > 0:
                self._logger.info(f"ベクトルストアの初期化が完了しました（{indexed_count}件の記事をインデックス化）")
                return vector_store
            else:
                self._logger.warning("インデックス化できた記事が0件でした")
                return None

        except Exception as e:
            self._logger.error(f"ベクトルストアの初期化に失敗しました: {e}", exc_info=True)
            return None

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
                                "text": "Gemmaモデルを使用したベクトル検索RAG対応チャットボットです。esa記事を意味的に検索して質問に答えます。",
                            },
                        },
                        {"type": "divider"},
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "💡 *使い方*\n• チャンネルでメンション: `@QAbot 質問内容`\n• DM: 直接メッセージを送信\n• スレッド: スレッド内で会話履歴を保持\n• ベクトル検索: esa記事を意味的に検索してRAG応答",
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
        """メッセージに対する応答を生成（RAG対応）"""
        # メンション記号を削除してクリーンなメッセージを取得
        cleaned_message = knowledge.clean_message(message_text)

        # Gemma Providerが使える場合
        if self._gemma_provider:
            try:
                # esa記事を検索してRAGで応答を生成
                context_docs = self._search_esa_documents(cleaned_message)

                if context_docs:
                    # esa記事が見つかった場合はRAGで応答生成
                    self._logger.info(f"esa記事 {len(context_docs)}件を使用してRAG応答を生成")
                    return self._gemma_provider.get_response_with_context(
                        thread_id,
                        cleaned_message,
                        context_docs
                    )
                else:
                    # esa記事が見つからない場合は通常の会話モード
                    self._logger.info("esa記事が見つからないため、通常の会話モードで応答")
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

    def _search_esa_documents(self, query: str) -> list[str]:
        """
        esa記事を検索し、本文を返す（ベクトル検索 or キーワード検索）

        Args:
            query: 検索クエリ

        Returns:
            記事の本文リスト
        """
        # ベクトル検索を使用
        if self._vector_store and self._vector_store.is_ready():
            try:
                # ベクトル検索で類似記事を取得
                search_results = self._vector_store.search(
                    query,
                    top_k=self._settings.vector_search_top_k
                )

                if not search_results:
                    self._logger.info("ベクトル検索で記事が見つかりませんでした")
                    return []

                # 検索結果を整形
                documents = []
                max_chars = self._settings.esa_max_chars_per_article
                for result in search_results:
                    post = result.post
                    score = result.score
                    # 記事の本文が長すぎる場合は切り詰める
                    body = post.body_md[:max_chars] if len(post.body_md) > max_chars else post.body_md
                    doc = f"タイトル: {post.title}\nURL: {post.url}\n類似度: {score:.3f}\n\n{body}"
                    documents.append(doc)

                self._logger.info(f"ベクトル検索で{len(documents)}件の記事を取得（最高スコア: {search_results[0].score:.3f}）")
                return documents

            except Exception as e:
                self._logger.error(f"ベクトル検索中にエラーが発生: {e}", exc_info=True)
                # フォールバック: キーワード検索を試行
                self._logger.info("キーワード検索にフォールバックします")

        # キーワード検索を使用（ベクトル検索が使えない場合）
        if not self._esa_client:
            return []

        try:
            max_articles = self._settings.esa_max_articles
            posts = self._esa_client.search_posts(query, per_page=max_articles)

            if not posts:
                return []

            documents = []
            max_chars = self._settings.esa_max_chars_per_article
            for post in posts:
                body = post.body_md[:max_chars] if len(post.body_md) > max_chars else post.body_md
                doc = f"タイトル: {post.title}\nURL: {post.url}\n\n{body}"
                documents.append(doc)

            self._logger.info(f"キーワード検索で{len(documents)}件の記事を取得")
            return documents

        except EsaClientError as e:
            self._logger.error(f"esa記事の検索に失敗: {e}", exc_info=True)
            return []
