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
2. Slackアプリのトークンを取得:
   - [Slack API](https://api.slack.com/apps) の「Your Apps」ページにアクセス
   - 使用するアプリの**アプリ名**（例: 「Lambda notify」）をクリックして、アプリの詳細設定ページに移動
   - **SLACK_BOT_TOKEN** (`xoxb-` で始まるトークン):
     - 左メニューの「**OAuth & Permissions**」をクリック
     - ページ上部の「**Bot User OAuth Token**」セクションを確認
     - 「**xoxb-**」で始まるトークンを「**Copy**」ボタンでコピー
     - もしトークンが表示されていない場合は、ページ上部の「**Install to Workspace**」ボタンをクリックしてワークスペースにインストール
   - **SLACK_SIGNING_SECRET**:
     - 左メニューの「**Basic Information**」をクリック
     - 「**App Credentials**」セクションまでスクロール
     - 「**Signing Secret**」の横にある「**Show**」をクリックして表示
     - 表示されたシークレットをコピー
   - **SLACK_APP_TOKEN** (`xapp-` で始まるトークン):
     - 左メニューの「**Basic Information**」をクリック
     - 「**App-Level Tokens**」セクションまでスクロール
     - 「**Generate Token and Scopes**」ボタンをクリック
     - トークン名を入力（例: 「Socket Mode」）
     - スコープに「**connections:write**」を追加
     - 「**Generate**」をクリックしてトークンを作成
     - 表示された「**xapp-**」で始まるトークンをコピー（この画面でしか表示されないため注意）
3. `.env` を作成:
   ```bash
   cp .env.example .env
   # 取得したトークンを設定
   ```

## 起動方法

```bash
uv run python main.py
```

起動後、SlackでBotにメンションすると登録済みキーワード（研究テーマ / メンバー構成 / ミーティング / 設備 / 募集）から最適な回答を返します。該当キーワードがない場合は `LAB_DEFAULT_RESPONSE` のメッセージを返します。

## FAQのカスタマイズ

`qabot/knowledge.py` の `DEFAULT_ENTRIES` を編集すると、キーワードと回答を自由に差し替えできます。`keywords` のタプルに検索したい語句を追加し、`answer` を研究室向けの文章に更新してください。`python main.py` を再起動すると即時反映されます。
