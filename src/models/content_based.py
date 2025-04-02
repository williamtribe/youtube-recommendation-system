from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import logging

class ContentBasedRecommender:
    """콘텐츠 기반 추천 시스템"""
    
    def __init__(self, feature_size: int = 776):
        """
        초기화 함수
        
        Args:
            feature_size (int): 입력 특징 벡터의 크기
        """
        self.logger = logging.getLogger(__name__)
        self.feature_size = feature_size
        self.video_features = {}
        self.video_ids = []
        
    def add_video(self, video_id: str, features: np.ndarray):
        """
        비디오와 특징을 추가
        
        Args:
            video_id (str): 비디오 ID
            features (np.ndarray): 특징 벡터
        """
        try:
            self.video_features[video_id] = features
            if video_id not in self.video_ids:
                self.video_ids.append(video_id)
        except Exception as e:
            self.logger.error(f"비디오 추가 중 오류 발생: {str(e)}")
    
    def get_similar_videos(self, video_id: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        유사한 비디오 추천
        
        Args:
            video_id (str): 기준 비디오 ID
            n_recommendations (int): 추천할 비디오 수
            
        Returns:
            List[Tuple[str, float]]: (비디오 ID, 유사도) 쌍의 리스트
        """
        try:
            if video_id not in self.video_features:
                raise KeyError(f"비디오를 찾을 수 없습니다: {video_id}")
            
            # 기준 비디오의 특징
            query_features = self.video_features[video_id]
            
            similarities = []
            for vid in self.video_ids:
                if vid != video_id:
                    similarity = cosine_similarity(
                        query_features.reshape(1, -1),
                        self.video_features[vid].reshape(1, -1)
                    )[0][0]
                    similarities.append((vid, similarity))
            
            # 유사도 기준 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:n_recommendations]
            
        except Exception as e:
            self.logger.error(f"유사 비디오 검색 중 오류 발생: {str(e)}")
            return []

    def batch_recommendations(self, video_ids: List[str], 
                            n_recommendations: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        여러 비디오에 대한 일괄 추천
        
        Args:
            video_ids (List[str]): 비디오 ID 리스트
            n_recommendations (int): 각 비디오당 추천 수
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: 비디오별 추천 결과
        """
        try:
            recommendations = {}
            for video_id in video_ids:
                recommendations[video_id] = self.get_similar_videos(
                    video_id, n_recommendations
                )
            return recommendations
            
        except Exception as e:
            self.logger.error(f"일괄 추천 중 오류 발생: {str(e)}")
            return {}