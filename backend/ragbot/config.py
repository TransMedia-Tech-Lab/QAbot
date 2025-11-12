"""設定値のデフォルト定義（パラメータはpyファイルに書く）"""

# RAG / Vector DB設定
DEFAULT_CHROMA_PERSIST_DIRECTORY = "./chroma_db"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# ログ設定
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE_BOT = "./logs/bot.log"
DEFAULT_LOG_FILE_SYNC = "./logs/sync.log"

# LLM設定
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"

