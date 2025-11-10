"""
esa APIクライアント
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger
import time


class EsaClient:
    """esa APIとの通信を担当するクライアント"""
    
    def __init__(self, access_token: str, team_name: str):
        self.access_token = access_token
        self.team_name = team_name
        self.base_url = f"https://api.esa.io/v1/teams/{team_name}"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
    def get_all_posts(self, updated_after: Optional[datetime] = None) -> List[Dict]:
        """
        全記事を取得（ページネーション対応）
        
        Args:
            updated_after: この日時以降に更新された記事のみ取得
            
        Returns:
            記事のリスト
        """
        all_posts = []
        page = 1
        per_page = 100
        
        while True:
            params = {
                "page": page,
                "per_page": per_page,
                "sort": "updated",
                "order": "desc"
            }
            
            if updated_after:
                params["q"] = f"updated:>={updated_after.isoformat()}"
            
            try:
                response = requests.get(
                    f"{self.base_url}/posts",
                    headers=self.headers,
                    params=params
                )
                response.raise_for_status()
                
                data = response.json()
                posts = data.get("posts", [])
                all_posts.extend(posts)
                
                logger.info(f"取得済み: {len(all_posts)}/{data.get('total_count')} 記事")
                
                # 次のページがない場合は終了
                if len(posts) < per_page or page >= data.get("total_pages", 1):
                    break
                    
                page += 1
                time.sleep(0.5)  # API制限対策
                
            except requests.exceptions.RequestException as e:
                logger.error(f"esa API エラー: {e}")
                break
                
        return all_posts
    
    def get_post(self, post_number: int) -> Optional[Dict]:
        """
        特定の記事を取得
        
        Args:
            post_number: 記事番号
            
        Returns:
            記事データ
        """
        try:
            response = requests.get(
                f"{self.base_url}/posts/{post_number}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"記事取得エラー (#{post_number}): {e}")
            return None
    
    def search_posts(self, query: str) -> List[Dict]:
        """
        記事を検索
        
        Args:
            query: 検索クエリ
            
        Returns:
            検索結果の記事リスト
        """
        try:
            response = requests.get(
                f"{self.base_url}/posts",
                headers=self.headers,
                params={"q": query}
            )
            response.raise_for_status()
            return response.json().get("posts", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"検索エラー: {e}")
            return []
