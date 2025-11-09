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
        use_quantization: True で 4bit量子化 (NF4)
        device: "cpu" または "cuda"
        """
        print(f"モデル '{model_name}' をロード中...")
        print(f"使用デバイス: {device}")
        print(f"量子化: {'有効' if use_quantization else '無効'}")
        print(f"PyTorchバージョン: {torch.__version__}")

        # PyTorch 2.0以降でtorch.compile()が利用可能
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version >= (2, 0):
            print("torch.compile()による最適化: 有効\n")
        else:
            print("torch.compile()による最適化: 無効 (PyTorch 2.0以降が必要)\n")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # pad_tokenを設定（eos_tokenと同じに設定）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 量子化の設定（メモリ削減）
        if use_quantization and device == "cuda":
            print("4bit量子化 (NF4) を適用中...\n")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
            # torch.compile()で推論を最適化
            try:
                print("モデルをコンパイル中（初回実行時は時間がかかります）...\n")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("モデルのコンパイルが完了しました\n")
            except Exception as e:
                print(f"モデルのコンパイルに失敗しました（通常モードで続行）: {e}\n")
        elif device == "cuda":
            # 量子化なし、GPU使用
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            # torch.compile()で推論を最適化
            try:
                print("モデルをコンパイル中（初回実行時は時間がかかります）...\n")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("モデルのコンパイルが完了しました\n")
            except Exception as e:
                print(f"モデルのコンパイルに失敗しました（通常モードで続行）: {e}\n")
        else:
            # CPU使用（bfloat16で高速化、float32より軽量）
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
                print("bfloat16を使用してモデルをロードしました\n")
            except Exception as e:
                print(f"bfloat16でのロードに失敗、float32にフォールバック: {e}\n")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
            # torch.compile()で推論を最適化
            try:
                print("モデルをコンパイル中（初回実行時は時間がかかります）...\n")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("モデルのコンパイルが完了しました\n")
            except Exception as e:
                print(f"モデルのコンパイルに失敗しました（通常モードで続行）: {e}\n")

        self.conversation_history = []
        print("モデル準備完了！\n")
    
    def chat(self, user_message: str, max_new_tokens: int = 256) -> str:
        """ユーザーのメッセージに対してGemmaが返答する"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            # Gemmaの公式chat_templateを使用（より最適化されている）
            chat_text = self.tokenizer.apply_chat_template(
                self.conversation_history,
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
            # torch.inference_mode()はtorch.no_grad()より高速
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    num_beams=1,  # ビームサーチなし（サンプリング使用時は1）
                    use_cache=True,  # KVキャッシュを有効化
                    repetition_penalty=1.1,  # 繰り返しを減らす
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # デコード（入力部分を除く）
            response_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

            # 会話履歴に追加
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            return response

        except Exception as e:
            return f"エラーが発生しました: {str(e)}"
    
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
    print("1. CPU (推奨)")
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