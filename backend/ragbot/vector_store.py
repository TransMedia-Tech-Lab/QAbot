"""
ベクトルデータベースとRAG検索エンジン
"""
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json
import re

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger
import tiktoken


class VectorStore:
    """ChromaDBを使用したベクトルストア"""
    
    def __init__(self, persist_directory: str, embedding_model: str):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model

        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 埋め込みモデルの初期化
        logger.info(f"埋め込みモデルをロード中: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        
        # ChromaDBクライアントの初期化
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # コレクションの作成または取得
        try:
            self.collection = self.client.get_collection("esa_articles")
            logger.info("既存のコレクションを使用")
        except:
            self.collection = self.client.create_collection(
                name="esa_articles",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("新規コレクションを作成")
    
    def _clean_metadata(self, metadata: Dict) -> Dict:
        """
        メタデータからNone値を除去（ChromaDBはNoneを許可しない）
        
        Args:
            metadata: メタデータ辞書
            
        Returns:
            クリーンアップされたメタデータ
        """
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                # Noneの場合は空文字列または0に変換
                if isinstance(key, str) and key.endswith("_index"):
                    cleaned[key] = 0
                else:
                    cleaned[key] = ""
            else:
                cleaned[key] = value
        return cleaned
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        ドキュメントをベクトルDBに追加
        
        Args:
            documents: esaの記事リスト
        """
        if not documents:
            return
        
        chunks = []
        for doc in documents:
            # 記事をチャンクに分割
            doc_chunks = self._split_document(doc)
            chunks.extend(doc_chunks)
        
        if not chunks:
            logger.warning("追加するチャンクがありません")
            return
        
        # バッチ処理で埋め込みとインデックス化
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            texts = [chunk["text"] for chunk in batch]
            # メタデータからNone値を除去
            metadatas = [self._clean_metadata(chunk["metadata"]) for chunk in batch]
            ids = [chunk["id"] for chunk in batch]
            
            # 埋め込みベクトルの生成
            embeddings = self.encoder.encode(texts).tolist()
            
            # ChromaDBに追加
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"チャンクを追加: {i + len(batch)}/{len(chunks)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        類似度検索を実行
        
        Args:
            query: 検索クエリ
            top_k: 取得する上位結果数
            
        Returns:
            検索結果のリスト
        """
        # クエリの埋め込み
        query_embedding = self.encoder.encode([query]).tolist()[0]
        
        # 検索実行
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 結果を整形
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] if "distances" in results else None
                search_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                    "similarity": (1 - distance) if distance is not None else None,
                })
        
        return search_results
    
    def update_document(self, document: Dict) -> None:
        """
        既存のドキュメントを更新
        
        Args:
            document: 更新する記事
        """
        # 既存のチャンクを削除
        post_number = document.get("number")
        if post_number:
            self.delete_document(post_number)
        
        # 新しいチャンクを追加
        self.add_documents([document])
    
    def delete_document(self, post_number: int) -> None:
        """
        ドキュメントを削除
        
        Args:
            post_number: 削除する記事番号
        """
        # 該当する全チャンクのIDを取得して削除
        chunk_ids = [f"post_{post_number}_chunk_{i}" for i in range(100)]  # 最大100チャンク
        try:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"記事 #{post_number} を削除しました")
        except Exception as e:
            logger.error(f"削除エラー: {e}")
    
    def _split_document(self, document: Dict) -> List[Dict]:
        """
        ドキュメントをチャンクに分割
        
        Args:
            document: esa記事
            
        Returns:
            チャンクのリスト
        """
        chunks = []
        post_number = document.get("number")
        title = document.get("name") or ""
        body = document.get("body_md") or ""
        url = document.get("url") or ""
        updated_at = document.get("updated_at") or ""
        created_by = document.get("created_by", {}).get("screen_name") or ""
        category = document.get("category") or ""
        
        # post_numberがNoneの場合はスキップ
        if post_number is None:
            logger.warning(f"記事番号がNoneのためスキップします: {document.get('name', 'Unknown')}")
            return chunks
        
        # 全体のテキストを作成
        full_text = f"# {title}\n\nカテゴリ: {category}\n\n{body}"
        
        # チャンク分割戦略：見出しベース
        sections = self._split_by_headers(full_text)
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            chunk_id = f"post_{post_number}_chunk_{i}"
            
            # ChromaDBはメタデータにNoneを許可しないため、Noneを空文字列に変換
            chunks.append({
                "id": chunk_id,
                "text": section,
                "metadata": {
                    "post_number": post_number,
                    "title": title or "",
                    "url": url or "",
                    "updated_at": updated_at or "",
                    "created_by": created_by or "",
                    "category": category or "",
                    "chunk_index": i
                }
            })
        
        return chunks
    
    def _split_by_headers(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        見出しベースでテキストを分割
        
        Args:
            text: 分割するテキスト
            max_chunk_size: 最大チャンクサイズ（文字数）
            
        Returns:
            チャンクのリスト
        """
        # 見出しパターン
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            
            # 見出しの場合
            if re.match(header_pattern, line):
                # 現在のチャンクが大きい場合は新しいチャンクを開始
                if current_size > max_chunk_size / 2:
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    current_chunk.append(line)
                    current_size += line_size
            else:
                # 通常のテキスト
                if current_size + line_size > max_chunk_size:
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    current_chunk.append(line)
                    current_size += line_size
        
        # 残りのチャンクを追加
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks


class RAGEngine:
    """RAG（Retrieval-Augmented Generation）エンジン"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # 類似度の最低閾値（cos類似度）
        self.min_similarity = float(os.getenv("RAG_MIN_SIMILARITY", "0.35"))
    
    def search_and_rank(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        ハイブリッド検索（ベクトル検索 + キーワードマッチング）
        
        Args:
            query: 検索クエリ
            top_k: 取得する上位結果数
            
        Returns:
            ランク付けされた検索結果
        """
        # ベクトル検索
        vector_results = self.vector_store.search(query, top_k * 2)
        
        # キーワード強調によるリランキング
        ranked_results = self._rerank_results(query, vector_results)
        
        # 類似度閾値でフィルタ
        filtered_results = self._filter_by_similarity(ranked_results)
        
        return filtered_results[:top_k]
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        検索結果をリランキング
        
        Args:
            query: 検索クエリ
            results: 初期検索結果
            
        Returns:
            リランキング後の結果
        """
        # クエリからキーワードを抽出
        keywords = self._extract_keywords(query)
        
        for result in results:
            text = result["text"].lower()
            score = result.get("distance", 0)
            result["similarity"] = result.get("similarity", (1 - score) if score is not None else None)
            
            # キーワードマッチングによるスコア調整
            keyword_boost = 0
            for keyword in keywords:
                if keyword.lower() in text:
                    keyword_boost += 0.1
            
            # タイトルマッチングによるブースト
            title = result["metadata"].get("title", "").lower()
            if any(kw.lower() in title for kw in keywords):
                keyword_boost += 0.2
            
            result["final_score"] = score - keyword_boost
        
        # スコアでソート（昇順）
        ranked_results = sorted(results, key=lambda x: x["final_score"])
        
        return ranked_results

    def _filter_by_similarity(self, results: List[Dict]) -> List[Dict]:
        """
        類似度閾値で結果をフィルタ
        """
        threshold = self.min_similarity
        if threshold <= 0:
            return results
        
        filtered = []
        for result in results:
            sim = result.get("similarity")
            if sim is None or sim >= threshold:
                filtered.append(result)
        
        if not filtered:
            logger.info("類似度閾値を満たす結果がありません (threshold=%.2f)", threshold)
        
        return filtered
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        クエリからキーワードを抽出
        
        Args:
            query: 検索クエリ
            
        Returns:
            キーワードのリスト
        """
        # 簡単なキーワード抽出（より高度な実装も可能）
        stopwords = {"は", "の", "を", "に", "が", "と", "で", "から", "まで", "より", "ですか", "ください"}
        
        # 形態素解析の代わりに簡易的な分割
        words = re.findall(r'[一-龥ぁ-んァ-ン\w]+', query)
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        
        return keywords
    
    def format_context(self, results: List[Dict], max_tokens: int = 2000) -> str:
        """
        検索結果をコンテキストとしてフォーマット
        
        Args:
            results: 検索結果
            max_tokens: 最大トークン数
            
        Returns:
            フォーマットされたコンテキスト
        """
        context_parts = []
        total_tokens = 0
        
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            text = result["text"]
            
            # メタデータを含むコンテキストを作成
            context_part = f"""
【参照{i}】
記事タイトル: {metadata.get('title', 'N/A')}
カテゴリ: {metadata.get('category', 'N/A')}
更新日: {metadata.get('updated_at', 'N/A')}
内容:
{text}
---"""
            
            # トークン数をチェック
            part_tokens = len(self.tokenizer.encode(context_part))
            if total_tokens + part_tokens > max_tokens:
                break
            
            context_parts.append(context_part)
            total_tokens += part_tokens
        
        return "\n".join(context_parts)
    
    def get_source_urls(self, results: List[Dict]) -> List[str]:
        """
        検索結果から参照URLを取得
        
        Args:
            results: 検索結果
            
        Returns:
            URLのリスト（重複削除済み）
        """
        urls = []
        seen_posts = set()
        
        for result in results:
            post_number = result["metadata"].get("post_number")
            url = result["metadata"].get("url")
            
            if post_number not in seen_posts and url:
                urls.append(url)
                seen_posts.add(post_number)
        
        return urls
