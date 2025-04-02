from typing import List, Dict, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import logging

class FeatureExtractor:
    """YouTube 비디오 특징 추출을 위한 클래스"""
    
    def __init__(self, bert_model_name: str = 'klue/bert-base'):
        """
        초기화 함수
        
        Args:
            bert_model_name (str): 사용할 BERT 모델 이름
        """
        self.logger = logging.getLogger(__name__)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        
    def extract_text_features(self, text: str) -> np.ndarray:
        """
        텍스트에서 BERT 임베딩 추출
        
        Args:
            text (str): 입력 텍스트
            
        Returns:
            np.ndarray: BERT 임베딩 벡터
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  max_length=512, truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
            return embeddings[0]
            
        except Exception as e:
            self.logger.error(f"텍스트 특징 추출 중 오류 발생: {str(e)}")
            return np.zeros(768)  # BERT 기본 임베딩 크기

    def extract_metadata_features(self, video_data: Dict) -> np.ndarray:
        """
        비디오 메타데이터에서 특징 추출
        
        Args:
            video_data (Dict): 비디오 메타데이터
            
        Returns:
            np.ndarray: 추출된 특징 벡터
        """
        try:
            features = []
            
            # 수치형 특징
            features.extend([
                float(video_data.get('view_count', 0)),
                float(video_data.get('like_count', 0)),
                float(video_data.get('comment_count', 0)),
                len(video_data.get('tags', [])),
                len(video_data.get('description', '')),
            ])
            
            # 정규화
            features = self.scaler.fit_transform(np.array(features).reshape(1, -1))[0]
            
            return features
            
        except Exception as e:
            self.logger.error(f"메타데이터 특징 추출 중 오류 발생: {str(e)}")
            return np.zeros(5)

    def extract_engagement_features(self, video_data: Dict) -> np.ndarray:
        """
        사용자 참여 관련 특징 추출
        
        Args:
            video_data (Dict): 비디오 메타데이터
            
        Returns:
            np.ndarray: 참여도 특징 벡터
        """
        try:
            views = float(video_data.get('view_count', 0))
            likes = float(video_data.get('like_count', 0))
            comments = float(video_data.get('comment_count', 0))
            
            features = []
            
            # 참여도 지표 계산
            if views > 0:
                features.extend([
                    likes / views,  # 좋아요 비율
                    comments / views,  # 댓글 비율
                    (likes + comments) / views  # 전체 참여도
                ])
            else:
                features.extend([0, 0, 0])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"참여도 특징 추출 중 오류 발생: {str(e)}")
            return np.zeros(3)

    def combine_features(self, text_features: np.ndarray, 
                        metadata_features: np.ndarray,
                        engagement_features: np.ndarray) -> np.ndarray:
        """
        모든 특징을 결합
        
        Args:
            text_features (np.ndarray): 텍스트 특징
            metadata_features (np.ndarray): 메타데이터 특징
            engagement_features (np.ndarray): 참여도 특징
            
        Returns:
            np.ndarray: 결합된 특징 벡터
        """
        try:
            combined = np.concatenate([
                text_features,
                metadata_features,
                engagement_features
            ])
            
            return combined
            
        except Exception as e:
            self.logger.error(f"특징 결합 중 오류 발생: {str(e)}")
            return np.zeros(776)  # 768 + 5 + 3

    def extract_all_features(self, video_data: Dict) -> Dict[str, np.ndarray]:
        """
        비디오에서 모든 특징 추출
        
        Args:
            video_data (Dict): 비디오 데이터
            
        Returns:
            Dict[str, np.ndarray]: 추출된 모든 특징
        """
        try:
            # 텍스트 특징 추출
            title_features = self.extract_text_features(video_data.get('title', ''))
            desc_features = self.extract_text_features(video_data.get('description', ''))
            
            # 메타데이터 및 참여도 특징 추출
            metadata_features = self.extract_metadata_features(video_data)
            engagement_features = self.extract_engagement_features(video_data)
            
            # 특징 결합
            combined_features = self.combine_features(
                (title_features + desc_features) / 2,  # 텍스트 특징 평균
                metadata_features,
                engagement_features
            )
            
            return {
                'text_features': (title_features + desc_features) / 2,
                'metadata_features': metadata_features,
                'engagement_features': engagement_features,
                'combined_features': combined_features
            }
            
        except Exception as e:
            self.logger.error(f"전체 특징 추출 중 오류 발생: {str(e)}")
            return {
                'text_features': np.zeros(768),
                'metadata_features': np.zeros(5),
                'engagement_features': np.zeros(3),
                'combined_features': np.zeros(776)
            }

    def save_features(self, features: Dict[str, np.ndarray], video_id: str):
        """
        추출된 특징을 파일로 저장
        
        Args:
            features (Dict[str, np.ndarray]): 저장할 특징들
            video_id (str): 비디오 ID
        """
        try:
            np.savez(
                f"data/features/{video_id}.npz",
                text_features=features['text_features'],
                metadata_features=features['metadata_features'],
                engagement_features=features['engagement_features'],
                combined_features=features['combined_features']
            )
            self.logger.info(f"특징이 성공적으로 저장되었습니다: {video_id}.npz")
        except Exception as e:
            self.logger.error(f"특징 저장 중 오류 발생: {str(e)}")