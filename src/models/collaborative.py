from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import torch
import torch.nn as nn
import logging

class CollaborativeRecommender:
    """협업 필터링 기반 추천 시스템"""
    
    def __init__(self, n_factors: int = 50):
        """
        초기화 함수
        
        Args:
            n_factors (int): 잠재 요인의 수
        """
        self.logger = logging.getLogger(__name__)
        self.n_factors = n_factors
        self.user_factors = None
        self.video_factors = None
        self.user_map = {}
        self.video_map = {}
        self.reverse_user_map = {}
        self.reverse_video_map = {}
        
    def _create_matrix(self, interactions: List[Dict]) -> csr_matrix:
        """
        상호작용 데이터로부터 희소 행렬 생성
        
        Args:
            interactions (List[Dict]): 사용자-비디오 상호작용 데이터
            
        Returns:
            csr_matrix: 사용자-비디오 상호작용 행렬
        """
        try:
            # 사용자와 비디오 ID 매핑
            user_ids = []
            video_ids = []
            ratings = []
            
            for interaction in interactions:
                user_id = interaction['user_id']
                video_id = interaction['video_id']
                rating = interaction.get('rating', 1.0)  # 기본값 1.0
                
                if user_id not in self.user_map:
                    idx = len(self.user_map)
                    self.user_map[user_id] = idx
                    self.reverse_user_map[idx] = user_id
                    
                if video_id not in self.video_map:
                    idx = len(self.video_map)
                    self.video_map[video_id] = idx
                    self.reverse_video_map[idx] = video_id
                
                user_ids.append(self.user_map[user_id])
                video_ids.append(self.video_map[video_id])
                ratings.append(rating)
            
            # 희소 행렬 생성
            matrix = csr_matrix(
                (ratings, (user_ids, video_ids)),
                shape=(len(self.user_map), len(self.video_map))
            )
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"행렬 생성 중 오류 발생: {str(e)}")
            return None

    def train(self, interactions: List[Dict]):
        """
        모델 학습
        
        Args:
            interactions (List[Dict]): 사용자-비디오 상호작용 데이터
        """
        try:
            # 상호작용 행렬 생성
            matrix = self._create_matrix(interactions)
            if matrix is None:
                return
            
            # SVD 수행
            U, sigma, Vt = svds(matrix, k=self.n_factors)
            
            # 사용자와 비디오 잠재 요인 저장
            self.user_factors = U
            self.video_factors = Vt.T
            
            self.logger.info("모델 학습이 완료되었습니다.")
            
        except Exception as e:
            self.logger.error(f"모델 학습 중 오류 발생: {str(e)}")

    def get_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        사용자별 추천 비디오 생성
        
        Args:
            user_id (str): 사용자 ID
            n_recommendations (int): 추천할 비디오 수
            
        Returns:
            List[Tuple[str, float]]: (비디오 ID, 예측 점수) 쌍의 리스트
        """
        try:
            if user_id not in self.user_map:
                raise KeyError(f"사용자를 찾을 수 없습니다: {user_id}")
            
            user_idx = self.user_map[user_id]
            user_vector = self.user_factors[user_idx]
            
            # 모든 비디오에 대한 예측 점수 계산
            predictions = np.dot(user_vector, self.video_factors.T)
            
            # 점수가 높은 순으로 정렬
            top_idxs = np.argsort(predictions)[-n_recommendations:][::-1]
            
            # 결과 변환
            recommendations = [
                (self.reverse_video_map[idx], float(predictions[idx]))
                for idx in top_idxs
            ]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"추천 생성 중 오류 발생: {str(e)}")
            return []

    def get_similar_videos(self, video_id: str, n_similar: int = 5) -> List[Tuple[str, float]]:
        """
        유사한 비디오 검색
        
        Args:
            video_id (str): 기준 비디오 ID
            n_similar (int): 검색할 유사 비디오 수
            
        Returns:
            List[Tuple[str, float]]: (비디오 ID, 유사도) 쌍의 리스트
        """
        try:
            if video_id not in self.video_map:
                raise KeyError(f"비디오를 찾을 수 없습니다: {video_id}")
            
            video_idx = self.video_map[video_id]
            video_vector = self.video_factors[video_idx]
            
            # 코사인 유사도 계산
            similarities = np.dot(self.video_factors, video_vector) / (
                np.linalg.norm(self.video_factors, axis=1) * np.linalg.norm(video_vector)
            )
            
            # 유사도가 높은 순으로 정렬
            top_idxs = np.argsort(similarities)[-n_similar-1:][::-1]
            
            # 자기 자신 제외
            top_idxs = top_idxs[top_idxs != video_idx][:n_similar]
            
            # 결과 변환
            similar_videos = [
                (self.reverse_video_map[idx], float(similarities[idx]))
                for idx in top_idxs
            ]
            
            return similar_videos
            
        except Exception as e:
            self.logger.error(f"유사 비디오 검색 중 오류 발생: {str(e)}")
            return []

    def save_model(self, filepath: str):
        """
        모델 저장
        
        Args:
            filepath (str): 저장할 파일 경로
        """
        try:
            model_data = {
                'user_factors': self.user_factors,
                'video_factors': self.video_factors,
                'user_map': self.user_map,
                'video_map': self.video_map,
                'reverse_user_map': self.reverse_user_map,
                'reverse_video_map': self.reverse_video_map
            }
            np.savez(filepath, **model_data)
            self.logger.info(f"모델이 저장되었습니다: {filepath}")
            
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")

    def load_model(self, filepath: str):
        """
        모델 로드
        
        Args:
            filepath (str): 로드할 파일 경로
        """
        try:
            model_data = np.load(filepath, allow_pickle=True)
            self.user_factors = model_data['user_factors']
            self.video_factors = model_data['video_factors']
            self.user_map = model_data['user_map'].item()
            self.video_map = model_data['video_map'].item()
            self.reverse_user_map = model_data['reverse_user_map'].item()
            self.reverse_video_map = model_data['reverse_video_map'].item()
            self.logger.info(f"모델을 로드했습니다: {filepath}")
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")