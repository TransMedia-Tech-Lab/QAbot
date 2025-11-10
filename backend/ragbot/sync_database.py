"""
定期同期スクリプト - cronやsystemdで実行
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

# パスの設定（親ディレクトリをパスに追加）
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from ragbot.esa_client import EsaClient
from ragbot.vector_store import VectorStore


def sync_database():
    """データベースを同期"""
    # 環境変数の読み込み
    load_dotenv()
    
    # ログ設定
    log_file = os.getenv("LOG_FILE", "./logs/sync.log")
    log_dir = os.path.dirname(log_file) or "."
    os.makedirs(log_dir, exist_ok=True)
    
    logger.add(
        log_file,
        rotation="1 week",
        retention="1 month",
        level="INFO"
    )
    
    logger.info("=" * 50)
    logger.info(f"同期開始: {datetime.now()}")
    
    try:
        # esaクライアントの初期化
        esa_client = EsaClient(
            access_token=os.environ.get("ESA_ACCESS_TOKEN") or os.environ.get("ESA_API_TOKEN"),
            team_name=os.environ.get("ESA_TEAM_NAME") or os.environ.get("ESA_TEAM"),
        )
        
        # ベクトルストアの初期化
        vector_store = VectorStore(
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        )
        
        # 最終同期時刻を取得（簡易実装）
        last_sync_file = ".last_sync"
        last_sync = None
        
        if os.path.exists(last_sync_file):
            with open(last_sync_file, "r") as f:
                last_sync_str = f.read().strip()
                if last_sync_str:
                    last_sync = datetime.fromisoformat(last_sync_str)
                    logger.info(f"最終同期: {last_sync}")
        
        # 記事を取得
        if last_sync:
            # 差分更新
            logger.info("差分更新を実行します")
            posts = esa_client.get_all_posts(updated_after=last_sync)
        else:
            # 全件取得
            logger.info("全件取得を実行します")
            posts = esa_client.get_all_posts()
        
        logger.info(f"取得した記事数: {len(posts)}")
        
        if posts:
            # ベクトルDBを更新
            for post in posts:
                vector_store.update_document(post)
            
            logger.info(f"{len(posts)}件の記事を更新しました")
        else:
            logger.info("更新する記事はありませんでした")
        
        # 最終同期時刻を保存
        with open(last_sync_file, "w") as f:
            f.write(datetime.now().isoformat())
        
        logger.info(f"同期完了: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"同期エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sync_database()
