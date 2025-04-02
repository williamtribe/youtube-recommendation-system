from typing import List, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.preprocessing import StandardScaler
import logging

class YouTubeDataPreprocessor:
    """YouTube 데이터 전처리를 위한 클래스"""
    
    def __init__(self):
        """초기화 함수"""
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()

    def parse_duration(self, duration: str) -> int:
        """
        YouTube API의 duration 포맷(ISO 8601)을 초 단위로 변환
        
        Args:
            duration (str): ISO 8601 형식의 기간 문자열 (예: 'PT1H2M10S')
            
        Returns:
            int: 초 단위로 변환된 기간
        """
        try:
            hours = re.search(r'(\d+)H', duration)
            minutes = re.search(r'(\d+)M', duration)
            seconds = re.search(r'(\d+)S', duration)
            
            hours = int(hours.group(1)) if hours else 0
            minutes = int(minutes.group(1)) if minutes else 0
            seconds = int(seconds.group(1)) if seconds else 0
            
            return hours * 3600 + minutes * 60 + seconds
        except Exception as e:
            self.logger.error(f"기간 파싱 중 오류 발생: {str(e)}")
            return 0

    def process_video_data(self, videos: List[Dict]) -> pd.DataFrame:
        """
        비디오 데이터 전처리
        
        Args:
            videos (List[Dict]): 비디오 데이터 리스트
            
        Returns:
            pd.DataFrame: 전처리된 비디오 데이터
        """
        try:
            df = pd.DataFrame(videos)
            
            # 날짜/시간 처리
            df['published_at'] = pd.to_datetime(df['published_at'])
            df['days_since_published'] = (datetime.now() - df['published_at']).dt.days
            
            # 기간 처리
            df['duration_seconds'] = df['duration'].apply(self.parse_duration)
            
            # 수치형 데이터 정규화
            numeric_columns = ['view_count', 'like_count', 'comment_count', 'duration_seconds']
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
            
            # 참여율 계산
            df['engagement_rate'] = (df['like_count'] + df['comment_count']) / df['view_count']
            
            return df
            
        except Exception as e:
            self.logger.error(f"비디오 데이터 처리 중 오류 발생: {str(e)}")
            return pd.DataFrame()

    def process_comments(self, comments: List[Dict]) -> pd.DataFrame:
        """
        댓글 데이터 전처리
        
        Args:
            comments (List[Dict]): 댓글 데이터 리스트
            
        Returns:
            pd.DataFrame: 전처리된 댓글 데이터
        """
        try:
            df = pd.DataFrame(comments)
            
            # 날짜/시간 처리
            df['published_at'] = pd.to_datetime(df['published_at'])
            df['days_since_published'] = (datetime.now() - df['published_at']).dt.days
            
            # 텍스트 전처리
            df['text_length'] = df['text'].str.len()
            df['word_count'] = df['text'].str.split().str.len()
            
            # 수치형 데이터 정규화
            numeric_columns = ['like_count', 'text_length', 'word_count']
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
            
            return df
            
        except Exception as e:
            self.logger.error(f"댓글 데이터 처리 중 오류 발생: {str(e)}")
            return pd.DataFrame()

    def extract_video_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        비디오 특징 추출
        
        Args:
            df (pd.DataFrame): 전처리된 비디오 데이터
            
        Returns:
            pd.DataFrame: 특징이 추출된 데이터
        """
        try:
            features = pd.DataFrame()
            
            # 기본 특징
            features['video_id'] = df['video_id']
            features['duration'] = df['duration_seconds']
            features['views'] = df['view_count']
            features['engagement'] = df['engagement_rate']
            features['days_online'] = df['days_since_published']
            
            # 일일 평균 지표
            features['daily_views'] = df['view_count'] / df['days_since_published']
            features['daily_likes'] = df['like_count'] / df['days_since_published']
            features['daily_comments'] = df['comment_count'] / df['days_since_published']
            
            return features
            
        except Exception as e:
            self.logger.error(f"특징 추출 중 오류 발생: {str(e)}")
            return pd.DataFrame()

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        전처리된 데이터를 파일로 저장
        
        Args:
            df (pd.DataFrame): 저장할 데이터프레임
            filename (str): 저장할 파일 이름
        """
        try:
            df.to_csv(f"data/processed/{filename}.csv", index=False, encoding='utf-8')
            self.logger.info(f"전처리된 데이터가 성공적으로 저장되었습니다: {filename}.csv")
        except Exception as e:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(e)}")