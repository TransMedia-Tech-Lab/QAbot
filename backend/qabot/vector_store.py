"""Vector store for semantic search using FAISS and sentence-transformers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .esa import EsaClient, EsaClientError, EsaPost


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """検索結果を表すデータクラス"""
    post: EsaPost
    score: float  # 類似度スコア（0-1、高いほど類似）


class EsaVectorStore:
    """esa記事のベクトル検索ストア"""

    def __init__(
        self,
        esa_client: EsaClient,
        embedding_model_name: str = "intfloat/multilingual-e5-base",
        device: str = "cpu"
    ):
        """
        ベクトルストアを初期化

        Args:
            esa_client: esa.io APIクライアント
            embedding_model_name: 使用するエンベディングモデル名
            device: 使用デバイス（cpu or cuda）
        """
        self._esa_client = esa_client
        self._logger = logging.getLogger(__name__)

        self._logger.info(f"エンベディングモデル '{embedding_model_name}' をロード中...")
        self._embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self._logger.info("エンベディングモデルのロードが完了しました")

        self._posts: List[EsaPost] = []
        self._index: Optional[faiss.Index] = None
        self._is_indexed = False

    def build_index(self, max_posts: int = 500) -> int:
        """
        esa記事を取得してインデックスを構築

        Args:
            max_posts: 取得する記事の最大数

        Returns:
            インデックス化された記事数
        """
        self._logger.info(f"esa記事を取得してインデックスを構築中（最大{max_posts}件）...")

        try:
            # esa記事を全件取得（ページネーション対応）
            posts = self._fetch_all_posts(max_posts)

            if not posts:
                self._logger.warning("取得できたesa記事が0件でした")
                return 0

            self._posts = posts
            self._logger.info(f"{len(posts)}件の記事を取得しました")

            # 各記事のテキストを作成（タイトル + 本文）
            texts = []
            for post in posts:
                # タイトルと本文を結合（タイトルは重要なので2回含める）
                text = f"{post.title} {post.title} {post.body_md}"
                texts.append(text)

            # テキストをベクトル化
            self._logger.info("記事をベクトル化中...")
            embeddings = self._embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # コサイン類似度用に正規化
            )

            # FAISSインデックスを構築
            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)  # Inner Product (コサイン類似度)
            self._index.add(embeddings)

            self._is_indexed = True
            self._logger.info(f"インデックスの構築が完了しました（{len(posts)}件、次元数: {dimension}）")

            return len(posts)

        except Exception as e:
            self._logger.error(f"インデックス構築中にエラーが発生: {e}", exc_info=True)
            return 0

    def _fetch_all_posts(self, max_posts: int) -> List[EsaPost]:
        """
        esa記事を全件取得（ページネーション対応）

        Args:
            max_posts: 取得する記事の最大数

        Returns:
            取得した記事のリスト
        """
        all_posts = []
        page = 1
        per_page = 100  # esa APIの1ページあたりの最大件数

        try:
            while len(all_posts) < max_posts:
                # 全記事を取得（更新日順）
                posts = self._esa_client.get_all_posts(per_page=per_page, page=page)

                if not posts:
                    break

                all_posts.extend(posts)
                self._logger.info(f"ページ {page}: {len(posts)}件取得（累計: {len(all_posts)}件）")

                if len(posts) < per_page:
                    # 最後のページに到達
                    break

                page += 1

            # 最大件数まで切り詰め
            return all_posts[:max_posts]

        except EsaClientError as e:
            self._logger.error(f"esa記事の取得に失敗: {e}", exc_info=True)
            return all_posts

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        クエリに類似した記事を検索

        Args:
            query: 検索クエリ
            top_k: 返す記事の最大数

        Returns:
            検索結果のリスト（類似度順）
        """
        if not self._is_indexed or self._index is None:
            self._logger.warning("インデックスが構築されていません")
            return []

        try:
            # クエリをベクトル化
            query_embedding = self._embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # FAISS検索
            scores, indices = self._index.search(query_embedding, top_k)

            # 結果を整形
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self._posts):
                    results.append(SearchResult(
                        post=self._posts[idx],
                        score=float(score)
                    ))

            max_score = results[0].score if results else 0.0
            self._logger.info(
                f"ベクトル検索完了: クエリ='{query[:50]}...', "
                f"結果={len(results)}件, "
                f"最高スコア={max_score:.3f}"
            )

            return results

        except Exception as e:
            self._logger.error(f"ベクトル検索中にエラーが発生: {e}", exc_info=True)
            return []

    def is_ready(self) -> bool:
        """インデックスが構築済みで検索可能かどうか"""
        return self._is_indexed and self._index is not None

    def get_indexed_count(self) -> int:
        """インデックス化された記事数を返す"""
        return len(self._posts) if self._is_indexed else 0
