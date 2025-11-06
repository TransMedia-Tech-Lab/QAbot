## QAbot Workspace

このリポジトリは Slack 研究室Botの **backend** と、今後拡張予定の **frontend** を分けて管理します。

- `backend/` : Slack Bolt を使ったBotアプリ（uv 仮想環境、Socket Mode対応）。
- `frontend/` : UIや設定画面を追加する場合のプレースホルダ。

### 使い方

1. `cd backend` して従来どおり `uv sync` / `uv run python main.py` を実行します。
2. `.env.example` も `backend/` 以下にあるので、そこでトークンを設定してください。
3. フロントエンドを追加したい場合は `frontend/` 内に任意のフレームワークで実装してください（現状は空ディレクトリ）。

バックエンドの詳しいセットアップやFAQカスタマイズ手順は `backend/README.md` を参照してください。

### コミットルール

- コミットメッセージは [Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/) に準拠して記述してください（例: `feat: add lab schedule intent`）。
- 1コミット1トピックを徹底し、不要なファイルや秘密情報（`.env` など）はステージングしないでください。
- feature ブランチで作業し、`main` への直接コミットは避けてください。
- 主なタイプ:
  - `feat`: 新しい機能
  - `fix`: バグ修正
  - `docs`: ドキュメントのみ
  - `style`: 空白やフォーマットなど
  - `refactor`: 挙動に影響しないリファクタ
  - `perf`: パフォーマンス改善
  - `test`: テスト追加・修正
  - `chore`: ビルド・ツール・依存管理
