"""
LLMラッパー（Ollama, Gemini対応）
"""
import os
from typing import Optional, Dict
from abc import ABC, abstractmethod
from loguru import logger

from .config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL


class LLMProvider(ABC):
    """LLMプロバイダの基底クラス"""
    
    @abstractmethod
    def generate(self, prompt: str, context: str) -> str:
        """回答を生成"""
        pass


class OllamaProvider(LLMProvider):
    """Ollamaを使用したローカルLLM"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
            self.model = model
            logger.info(f"Ollama初期化完了: {model}")
        except ImportError:
            logger.error("ollamaライブラリがインストールされていません")
            raise
    
    def generate(self, prompt: str, context: str) -> str:
        """
        Ollamaで回答を生成
        
        Args:
            prompt: ユーザーの質問
            context: 検索結果のコンテキスト
            
        Returns:
            生成された回答
        """
        system_prompt = """あなたは研究室の情報を提供するアシスタントです。
以下の参照情報を基に、正確で簡潔な回答を日本語で提供してください。
参照情報にない内容については推測せず、「情報が見つかりませんでした」と回答してください。
重要な情報（鍵番号、パスワード等）は正確に伝えてください。"""
        
        user_message = f"""以下の参照情報を基に質問に回答してください。

参照情報:
{context}

質問: {prompt}

回答:"""
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama生成エラー: {e}")
            return "申し訳ありません。回答の生成中にエラーが発生しました。"


class GeminiProvider(LLMProvider):
    """Google Gemini APIを使用"""
    
    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini API初期化完了")
        except ImportError:
            logger.error("google-generativeaiライブラリがインストールされていません")
            raise
    
    def generate(self, prompt: str, context: str) -> str:
        """
        Gemini APIで回答を生成
        
        Args:
            prompt: ユーザーの質問
            context: 検索結果のコンテキスト
            
        Returns:
            生成された回答
        """
        system_instruction = """あなたは研究室の情報を提供するアシスタントです。
参照情報を基に正確で簡潔な回答を日本語で提供してください。
参照情報にない内容は推測せず、「情報が見つかりませんでした」と回答してください。"""
        
        full_prompt = f"""{system_instruction}

参照情報:
{context}

質問: {prompt}

回答:"""
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API生成エラー: {e}")
            return "申し訳ありません。回答の生成中にエラーが発生しました。"


class LLMManager:
    """LLMプロバイダを管理するマネージャー"""
    
    def __init__(self):
        self.provider = self._initialize_provider()
    
    def _initialize_provider(self) -> LLMProvider:
        """環境変数に基づいてLLMプロバイダを初期化"""
        
        # Ollama（優先度1）
        if os.getenv("OLLAMA_MODEL"):
            logger.info("Ollamaプロバイダを使用します")
            return OllamaProvider(
                base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
                model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
            )
        
        # Gemini API（優先度2）
        if os.getenv("GEMINI_API_KEY"):
            logger.info("Gemini APIプロバイダを使用します")
            return GeminiProvider(api_key=os.getenv("GEMINI_API_KEY"))
        
        # デフォルト（Ollama）
        logger.warning("LLM設定が見つかりません。デフォルトでOllamaを使用します")
        return OllamaProvider()
    
    def generate_answer(self, question: str, context: str, metadata: Optional[Dict] = None) -> str:
        """
        質問に対する回答を生成
        
        Args:
            question: ユーザーの質問
            context: 検索結果のコンテキスト
            metadata: 追加のメタデータ
            
        Returns:
            生成された回答
        """
        # コンテキストが空の場合
        if not context or context.strip() == "":
            return "申し訳ありません。その質問に関連する情報が見つかりませんでした。"
        
        # プロンプトの強化
        enhanced_prompt = self._enhance_prompt(question, metadata)
        
        # 回答生成
        answer = self.provider.generate(enhanced_prompt, context)
        
        # 後処理
        answer = self._postprocess_answer(answer)
        
        return answer
    
    def _enhance_prompt(self, question: str, metadata: Optional[Dict] = None) -> str:
        """
        プロンプトを強化
        
        Args:
            question: 元の質問
            metadata: 追加情報
            
        Returns:
            強化されたプロンプト
        """
        enhanced = question
        
        # 質問の種類を判定して適切な指示を追加
        if "鍵" in question or "パスワード" in question:
            enhanced += "\n※セキュリティ情報は正確に伝えてください。"
        elif "研究室" in question and "どのような" in question:
            enhanced += "\n※研究室の特徴や研究分野を簡潔にまとめてください。"
        elif "誰" in question or "メンバー" in question:
            enhanced += "\n※人物情報は役職や研究テーマも含めて回答してください。"
        
        return enhanced
    
    def _postprocess_answer(self, answer: str) -> str:
        """
        生成された回答の後処理
        
        Args:
            answer: 生成された回答
            
        Returns:
            後処理済みの回答
        """
        # 不要な前置きを削除
        remove_phrases = [
            "参照情報によると、",
            "参照情報を基に回答します。",
            "以下が回答です："
        ]
        
        for phrase in remove_phrases:
            answer = answer.replace(phrase, "")
        
        # 改行の正規化
        answer = answer.strip()
        
        return answer
