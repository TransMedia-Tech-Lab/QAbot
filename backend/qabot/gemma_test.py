import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# シンボリックリンク警告を無視
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

class GemmaChatTransformers:
    def __init__(self, model_name: str = "google/gemma-3n-E2b-it", use_quantization: bool = True, device: str = "cpu"):
        """
        モデルとトークナイザーを初期化
        model_name: "google/gemma-3n-e2b-it"
        use_quantization: True で 8bit量子化
        device: "cpu" または "cuda"
        """
        print(f"モデル '{model_name}' をロード中...")
        print(f"使用デバイス: {device}")
        print(f"量子化: {'有効' if use_quantization else '無効'}\n")
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # pad_tokenを設定（eos_tokenと同じに設定）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 量子化の設定（メモリ削減）
        if use_quantization and device == "cuda":
            print("8bit量子化を適用中...\n")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=200.0,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="cuda",
                low_cpu_mem_usage=True
            )
        elif device == "cuda":
            # 量子化なし、GPU使用
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # CPU使用
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        
        self.conversation_history = []
        print("モデル準備完了！\n")
    
    def chat(self, user_message: str, max_new_tokens: int = 256) -> str:
        """ユーザーのメッセージに対してGemmaが返答する"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # 会話履歴をテキストに変換
        conversation_text = self._format_conversation()
        
        try:
            # トークン化
            inputs = self.tokenizer(
                conversation_text,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            ).to(self.device)
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # 生成
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # デコード（入力部分を除く）
            response_tokens = output_ids[0, input_ids.shape[1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # 会話履歴に追加
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            return response
        
        except Exception as e:
            return f"エラーが発生しました: {str(e)}"
    
    def _format_conversation(self) -> str:
        """会話履歴をモデル用のテキスト形式に変換"""
        formatted = ""
        for msg in self.conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            formatted += f"{role}: {msg['content']}\n"
        formatted += "model: "
        return formatted
    
    def clear_history(self):
        """会話履歴をクリア"""
        self.conversation_history = []
    
    def show_history(self):
        """会話履歴を表示"""
        print("\n--- 会話履歴 ---")
        for msg in self.conversation_history:
            role = "あなた" if msg["role"] == "user" else "Gemma"
            print(f"{role}: {msg['content']}\n")


def select_device():
    """ユーザーにデバイスを選択させる"""
    print("="*60)
    print("使用するデバイスを選択してください")
    print("="*60)
    print("1. CPU (推奨, 遅いが確実)")
    print("2. GPU (CUDA対応, 可能な場合)")
    
    while True:
        choice = input("\n選択 (1 or 2): ").strip()
        if choice == "1":
            return "cpu"
        elif choice == "2":
            if torch.cuda.is_available():
                print(f"GPU検出: {torch.cuda.get_device_name(0)}\n")
                return "cuda"
            else:
                print("CUDA対応GPUが検出されません。CPUを使用します\n")
                return "cpu"
        else:
            print("1 または 2 を入力してください")


def main():
    print("="*60)
    print("Gemma チャットボット(test版)")
    print("="*60 + "\n")
    
    device = select_device()
    
    try:
        chat = GemmaChatTransformers(
            use_quantization=True,  # メモリ削減のため常に有効
            device=device
        )
    except Exception as e:
        print(f"エラー: {e}")
        return
    
    print("="*60)
    print("コマンド:")
    print("  'quit' または 'exit' : 終了")
    print("  'clear' : 会話履歴をクリア")
    print("  'history' : 会話履歴を表示")
    print("="*60 + "\n")
    
    while True:
        user_input = input("あなた: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("終了します")
            break
        
        if user_input.lower() == 'clear':
            chat.clear_history()
            print("会話履歴をクリアしました\n")
            continue
        
        if user_input.lower() == 'history':
            chat.show_history()
            continue
        
        print("Gemmaが考え中...\n")
        response = chat.chat(user_input)
        print(f"Gemma: {response}\n")


if __name__ == "__main__":
    main()