## 概要

Slackでメンションすると研究室に関するFAQを返してくれる簡易Botです。Socket Modeで動作するため、社内公開は不要です。

## 必要なもの

- Slackアプリ（Bot Token Scopes 例: `app_mentions:read`, `chat:write`, `commands`, `im:history`）
- App-Level Token（`connections:write`）
- Bot User OAuth Token
- Python 3.13（`uv`が自動で `.venv` を作成）

## セットアップ

1. 依存関係をインストール:
   ```bash
   uv sync
   ```
2. `.env` を作成:
   ```bash
   cp .env.example .env
   # SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_SIGNING_SECRET を設定
   ```

## 起動方法

```bash
uv run python main.py
```

起動後、SlackでBotにメンションすると登録済みキーワード（研究テーマ / メンバー構成 / ミーティング / 設備 / 募集）から最適な回答を返します。該当キーワードがない場合は `LAB_DEFAULT_RESPONSE` のメッセージを返します。

## FAQのカスタマイズ

`qabot/knowledge.py` の `DEFAULT_ENTRIES` を編集すると、キーワードと回答を自由に差し替えできます。`keywords` のタプルに検索したい語句を追加し、`answer` を研究室向けの文章に更新してください。`python main.py` を再起動すると即時反映されます。
