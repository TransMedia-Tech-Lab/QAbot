## 概要

`backend/` には esa 記事を検索・同期し、ChromaDB + LLM で回答する Slack RAG ボットが含まれます。エントリポイントは `main.py` のみで、Slack Socket Mode に対応しています。

## ディレクトリ構成

```
backend/
├── main.py            # ResearchLabBot のエントリポイント
├── ragbot/            # bot実装・esaクライアント・RAG/LLMユーティリティ
│   ├── bot.py
│   ├── esa_client.py
│   ├── vector_store.py
│   ├── llm_manager.py
│   └── sync_database.py
├── chroma_db/         # ChromaDBの永続化ディレクトリ（.gitignore）
├── logs/              # loguru が出力するログ（.gitignore）
├── .env.example       # 必須環境変数のサンプル
├── pyproject.toml     # uv 用プロジェクト設定
└── uv.lock            # 依存関係ロックファイル
```

## セットアップ

1. 依存関係をインストール
   ```bash
   cd backend
   uv sync
   ```
2. 環境変数を設定
   ```bash
   cp .env.example .env
   # Slack / esa / LLM を記入
   ```
3. 初回起動時に自動で `chroma_db/` `logs/` が作成されます。必要に応じて事前に `mkdir -p chroma_db logs` しておいても構いません。

## 主要な環境変数

| 変数 | 用途 |
| --- | --- |
| `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_SIGNING_SECRET` | Slack 認証情報（必須） |
| `ESA_ACCESS_TOKEN`, `ESA_TEAM_NAME` | esa API へのアクセスに使用（`ESA_API_TOKEN` / `ESA_TEAM` でも可） |
| `OLLAMA_MODEL` | ローカル LLM を使用する場合に設定（優先度1） |
| `GEMINI_API_KEY` | Google Gemini API キー（優先度2） |
| `RAG_MIN_SIMILARITY` | RAGの類似度フィルタ閾値（cos類似度 0〜1、デフォルト0.35） |

**注意**: 設定値（`CHROMA_PERSIST_DIRECTORY`, `EMBEDDING_MODEL`, `LOG_LEVEL`, `LOG_FILE`, `OLLAMA_BASE_URL`など）は `ragbot/config.py` にデフォルト値として定義されています。必要に応じて環境変数で上書きできますが、通常は変更不要です。

詳細は `.env.example` を参照してください。

## 起動方法

```bash
uv run python main.py
```

- 初回起動時に esa から記事を取得して `chroma_db/` に保存します。
- Slackではメンション／DM／`/lab` コマンドに対応します。
  - `/lab search <質問>` … esa + RAG で検索し、回答と参照URLを返します。
  - `/lab sync` … esa→ChromaDB の同期を手動で実行します。
  - `/lab stats` … チャンク数や利用中のLLMなどを表示します。

CLIから直接同期したい場合は以下を実行してください。

```bash
uv run ragbot/sync_database.py
```

## 運用メモ

- `logs/bot.log` に日次ローテーションでアプリログが保存されます。障害解析や同期状況の監視に活用してください。
- `chroma_db/` はChromaDBの永続化ディレクトリです。バックアップや定期的なクリーンアップを検討してください。
- 依存関係・ドキュメントは `pyproject.toml` とこの `README.md` に集約されています。
