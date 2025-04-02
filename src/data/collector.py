from typing import List, Dict, Optional
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from datetime import datetime

class YouTubeDataCollector:
    """YouTube 데이터 수집을 위한 클래스"""
    
    def __init__(self, api_key: str):
        """
        초기화 함수
        
        Args:
            api_key (str): YouTube Data API 키
        """
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.logger = logging.getLogger(__name__)

    def collect_video_data(self, video_id: str) -> Dict:
        """
        비디오 메타데이터 수집
        
        Args:
            video_id (str): YouTube 비디오 ID
            
        Returns:
            Dict: 비디오 정보를 담은 딕셔너리
        """
        try:
            video_response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            ).execute()

            if not video_response['items']:
                self.logger.error(f"비디오를 찾을 수 없습니다: {video_id}")
                return {}

            video_data = video_response['items'][0]
            
            return {
                'video_id': video_id,
                'title': video_data['snippet']['title'],
                'description': video_data['snippet']['description'],
                'published_at': video_data['snippet']['publishedAt'],
                'channel_id': video_data['snippet']['channelId'],
                'channel_title': video_data['snippet']['channelTitle'],
                'tags': video_data['snippet'].get('tags', []),
                'category_id': video_data['snippet']['categoryId'],
                'duration': video_data['contentDetails']['duration'],
                'view_count': int(video_data['statistics'].get('viewCount', 0)),
                'like_count': int(video_data['statistics'].get('likeCount', 0)),
                'comment_count': int(video_data['statistics'].get('commentCount', 0))
            }
            
        except HttpError as e:
            self.logger.error(f"API 호출 중 오류 발생: {str(e)}")
            return {}

    def collect_comments(self, video_id: str, max_results: int = 100) -> List[Dict]:
        """
        비디오 댓글 수집
        
        Args:
            video_id (str): YouTube 비디오 ID
            max_results (int): 수집할 최대 댓글 수
            
        Returns:
            List[Dict]: 댓글 정보를 담은 리스트
        """
        try:
            comments = []
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(max_results, 100),
                textFormat='plainText'
            )

            while request and len(comments) < max_results:
                response = request.execute()
                
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'comment_id': item['id'],
                        'video_id': video_id,
                        'text': comment['textDisplay'],
                        'author': comment['authorDisplayName'],
                        'published_at': comment['publishedAt'],
                        'like_count': comment['likeCount']
                    })
                
                request = self.youtube.commentThreads().list_next(request, response)
                
            return comments
            
        except HttpError as e:
            self.logger.error(f"댓글 수집 중 오류 발생: {str(e)}")
            return []

    def collect_related_videos(self, video_id: str, max_results: int = 50) -> List[Dict]:
        """
        연관 비디오 수집
        
        Args:
            video_id (str): YouTube 비디오 ID
            max_results (int): 수집할 최대 연관 비디오 수
            
        Returns:
            List[Dict]: 연관 비디오 정보를 담은 리스트
        """
        try:
            request = self.youtube.search().list(
                part='snippet',
                relatedToVideoId=video_id,
                type='video',
                maxResults=min(max_results, 50)
            )
            
            response = request.execute()
            related_videos = []
            
            for item in response['items']:
                video = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'channel_id': item['snippet']['channelId'],
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail': item['snippet']['thumbnails']['default']['url']
                }
                related_videos.append(video)
            
            return related_videos
            
        except HttpError as e:
            self.logger.error(f"연관 비디오 수집 중 오류 발생: {str(e)}")
            return []

    def save_to_csv(self, data: List[Dict], filename: str):
        """
        데이터를 CSV 파일로 저장
        
        Args:
            data (List[Dict]): 저장할 데이터
            filename (str): 저장할 파일 이름
        """
        try:
            df = pd.DataFrame(data)
            df.to_csv(f"data/{filename}.csv", index=False, encoding='utf-8')
            self.logger.info(f"데이터가 성공적으로 저장되었습니다: {filename}.csv")
        except Exception as e:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(e)}")