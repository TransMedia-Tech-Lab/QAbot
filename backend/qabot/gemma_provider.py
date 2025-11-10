"""Gemma-based chat provider for Slack bot."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class GemmaChatTransformers:
    """Gemmaモデルを使用したチャット機能"""

    def __init__(self, model_name: str = "google/gemma-3n-e2b-it", device: str = "cpu"):
        """
        モデルとトークナイザーを初期化

        Args:
            model_name: Gemmaモデル名
            device: 使用デバイス（CPU固定を推奨）
        """
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"モデル '{model_name}' をロード中...")
        self._logger.info(f"使用デバイス: {device}")
        self._logger.info(f"PyTorchバージョン: {torch.__version__}")

        # シンボリックリンク警告を無視
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # pad_tokenを設定（eos_tokenと同じに設定）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CPU使用（bfloat16で高速化、float32より軽量）
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            self._logger.info("bfloat16を使用してモデルをロードしました")
        except Exception as e:
            self._logger.warning(f"bfloat16でのロードに失敗、float32にフォールバック: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )

        # torch.compile()で推論を最適化
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version >= (2, 0):
            try:
                self._logger.info("モデルをコンパイル中（初回実行時は時間がかかります）...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._logger.info("モデルのコンパイルが完了しました")
            except Exception as e:
                self._logger.warning(f"モデルのコンパイルに失敗しました（通常モードで続行）: {e}")

        self._logger.info("Gemmaモデル準備完了")

    def generate_response(
        self,
        conversation_history: List[Dict[str, str]],
        max_new_tokens: int = 256
    ) -> str:
        """
        会話履歴を基に応答を生成

        Args:
            conversation_history: 会話履歴（[{"role": "user"|"assistant", "content": "..."}]形式）
            max_new_tokens: 生成する最大トークン数

        Returns:
            生成された応答テキスト
        """
        try:
            # Gemmaの公式chat_templateを使用
            chat_text = self.tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True
            )

            # トークン化
            inputs = self.tokenizer(
                chat_text,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)

            # 生成（最適化されたパラメータ）
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    num_beams=1,
                    use_cache=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # デコード（入力部分を除く）
            response_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

            return response

        except Exception as e:
            self._logger.error(f"応答生成中にエラーが発生: {e}", exc_info=True)
            return f"エラーが発生しました: {str(e)}"


class GemmaAnswerProvider:
    """スレッドごとの会話履歴を管理し、Gemmaで応答を生成"""

    def __init__(self, model_name: str = "google/gemma-3n-e2b-it", device: str = "cpu"):
        """
        Gemma Answer Providerを初期化

        Args:
            model_name: Gemmaモデル名
            device: 使用デバイス（CPU固定を推奨）
        """
        self._logger = logging.getLogger(__name__)
        self._gemma_chat = GemmaChatTransformers(model_name=model_name, device=device)
        self._thread_histories: Dict[str, List[Dict[str, str]]] = {}
        self._logger.info("Gemma Answer Provider初期化完了")

    def get_response(self, thread_id: str, user_message: str, max_new_tokens: int = 256) -> str:
        """
        ユーザーメッセージに対する応答を生成

        Args:
            thread_id: スレッドID（会話履歴の識別子）
            user_message: ユーザーからのメッセージ
            max_new_tokens: 生成する最大トークン数

        Returns:
            Gemmaからの応答
        """
        # スレッドの会話履歴を取得または作成
        if thread_id not in self._thread_histories:
            self._thread_histories[thread_id] = []

        history = self._thread_histories[thread_id]

        # 一時的な履歴を作成（まだhistoryには追加しない）
        temp_history = history + [{
            "role": "user",
            "content": user_message
        }]

        try:
            # Gemmaで応答を生成
            response = self._gemma_chat.generate_response(temp_history, max_new_tokens)
        except Exception as e:
            self._logger.error(f"応答生成中にエラーが発生: {e}", exc_info=True)
            response = f"申し訳ございません。応答の生成中にエラーが発生しました: {str(e)}"

        # 成功・失敗に関わらず、ユーザーメッセージとアシスタントの応答を履歴に追加
        history.append({
            "role": "user",
            "content": user_message
        })
        history.append({
            "role": "assistant",
            "content": response
        })

        self._logger.info(f"スレッド {thread_id} に応答を生成（履歴: {len(history)}メッセージ）")

        return response

    def get_response_with_context(
        self,
        thread_id: str,
        user_message: str,
        context_documents: List[str],
        max_new_tokens: int = 512
    ) -> str:
        """
        コンテキスト（参考資料）を含めて応答を生成（RAG）

        Args:
            thread_id: スレッドID（会話履歴の識別子）
            user_message: ユーザーからのメッセージ
            context_documents: 参考資料のリスト（esa記事の本文など）
            max_new_tokens: 生成する最大トークン数

        Returns:
            Gemmaからの応答
        """
        # スレッドの会話履歴を取得または作成
        if thread_id not in self._thread_histories:
            self._thread_histories[thread_id] = []

        history = self._thread_histories[thread_id]

        # コンテキストを含むプロンプトを構築
        context_text = "\n\n".join([
            f"【参考資料{i+1}】\n{doc}" for i, doc in enumerate(context_documents)
        ])

        augmented_message = f"""以下の参考資料を基に、ユーザーの質問に答えてください。

{context_text}

【ユーザーの質問】
{user_message}

【回答ルール】
1. 参考資料の内容を基に、正確かつ簡潔に回答してください
2. 参考資料に情報がない場合は「参考資料には記載がありませんでした」と正直に答えてください
3. 参考資料を引用する際は、どの資料を参照したか明示してください
4. 日本語で回答してください"""

        # 拡張されたプロンプトで一時的な履歴を作成（まだhistoryには追加しない）
        temp_history = history + [{
            "role": "user",
            "content": augmented_message
        }]

        try:
            # Gemmaで応答を生成
            response = self._gemma_chat.generate_response(temp_history, max_new_tokens)
        except Exception as e:
            self._logger.error(f"RAG応答生成中にエラーが発生: {e}", exc_info=True)
            response = f"申し訳ございません。応答の生成中にエラーが発生しました: {str(e)}"

        # 成功・失敗に関わらず、ユーザーメッセージとアシスタントの応答を履歴に追加
        history.append({
            "role": "user",
            "content": user_message
        })
        history.append({
            "role": "assistant",
            "content": response
        })

        self._logger.info(
            f"スレッド {thread_id} にRAG応答を生成（コンテキスト: {len(context_documents)}件、履歴: {len(history)}メッセージ）"
        )

        return response

    def clear_thread_history(self, thread_id: str) -> None:
        """特定のスレッドの会話履歴をクリア"""
        if thread_id in self._thread_histories:
            del self._thread_histories[thread_id]
            self._logger.info(f"スレッド {thread_id} の履歴をクリアしました")

    def clear_all_histories(self) -> None:
        """全てのスレッドの会話履歴をクリア"""
        self._thread_histories.clear()
        self._logger.info("全スレッドの履歴をクリアしました")

    def get_thread_count(self) -> int:
        """管理中のスレッド数を取得"""
        return len(self._thread_histories)
