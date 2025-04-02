from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class FeatureAnalyzer:
    """YouTube 비디오 특징 분석을 위한 클래스"""
    
    def __init__(self):
        """초기화 함수"""
        self.logger = logging.getLogger(__name__)
        self.pca = PCA(n_components=50)
        self.tsne = TSNE(n_components=2)
        self.kmeans = KMeans(n_clusters=5)

    def reduce_dimensions(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        특징 차원 축소
        
        Args:
            features (np.ndarray): 원본 특징 벡터
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (PCA 결과, t-SNE 결과)
        """
        try:
            # PCA 적용
            pca_features = self.pca.fit_transform(features)
            
            # t-SNE 적용
            tsne_features = self.tsne.fit_transform(pca_features)
            
            return pca_features, tsne_features
            
        except Exception as e:
            self.logger.error(f"차원 축소 중 오류 발생: {str(e)}")
            return np.zeros((features.shape[0], 50)), np.zeros((features.shape[0], 2))

    def cluster_videos(self, features: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        """
        비디오 클러스터링
        
        Args:
            features (np.ndarray): 특징 벡터
            n_clusters (int): 클러스터 수
            
        Returns:
            np.ndarray: 클러스터 레이블
        """
        try:
            self.kmeans = KMeans(n_clusters=n_clusters)
            labels = self.kmeans.fit_predict(features)
            return labels
            
        except Exception as e:
            self.logger.error(f"클러스터링 중 오류 발생: {str(e)}")
            return np.zeros(features.shape[0])

    def analyze_clusters(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        클러스터 분석
        
        Args:
            features (np.ndarray): 특징 벡터
            labels (np.ndarray): 클러스터 레이블
            
        Returns:
            Dict: 클러스터 분석 결과
        """
        try:
            analysis = {}
            
            # 클러스터별 크기
            analysis['cluster_sizes'] = np.bincount(labels)
            
            # 클러스터별 중심점
            analysis['centroids'] = self.kmeans.cluster_centers_
            
            # 클러스터 내 분산
            analysis['inertia'] = self.kmeans.inertia_
            
            # 클러스터별 특징 평균
            cluster_means = []
            for i in range(len(np.unique(labels))):
                cluster_means.append(features[labels == i].mean(axis=0))
            analysis['cluster_means'] = np.array(cluster_means)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"클러스터 분석 중 오류 발생: {str(e)}")
            return {}

    def visualize_clusters(self, tsne_features: np.ndarray, labels: np.ndarray, 
                         save_path: str = 'visualizations/clusters.png'):
        """
        클러스터 시각화
        
        Args:
            tsne_features (np.ndarray): t-SNE로 축소된 특징
            labels (np.ndarray): 클러스터 레이블
            save_path (str): 저장할 파일 경로
        """
        try:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                                c=labels, cmap='viridis')
            plt.colorbar(scatter)
            plt.title('Video Clusters Visualization')
            plt.xlabel('t-SNE dimension 1')
            plt.ylabel('t-SNE dimension 2')
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"클러스터 시각화가 저장되었습니다: {save_path}")
            
        except Exception as e:
            self.logger.error(f"시각화 중 오류 발생: {str(e)}")

    def find_similar_videos(self, features: np.ndarray, query_features: np.ndarray, 
                          n_similar: int = 5) -> List[int]:
        """
        유사한 비디오 찾기
        
        Args:
            features (np.ndarray): 전체 비디오 특징
            query_features (np.ndarray): 쿼리 비디오 특징
            n_similar (int): 찾을 유사 비디오 수
            
        Returns:
            List[int]: 유사한 비디오의 인덱스 목록
        """
        try:
            # 코사인 유사도 계산
            similarities = np.dot(features, query_features) / (
                np.linalg.norm(features, axis=1) * np.linalg.norm(query_features)
            )
            
            # 가장 유사한 비디오의 인덱스 반환
            similar_indices = np.argsort(similarities)[-n_similar:][::-1]
            return similar_indices.tolist()
            
        except Exception as e:
            self.logger.error(f"유사 비디오 검색 중 오류 발생: {str(e)}")
            return []

    def analyze_feature_importance(self, features: np.ndarray) -> pd.DataFrame:
        """
        특징 중요도 분석
        
        Args:
            features (np.ndarray): 특징 벡터
            
        Returns:
            pd.DataFrame: 특징 중요도 분석 결과
        """
        try:
            # PCA 성분의 설명 분산 비율 계산
            explained_variance_ratio = self.pca.explained_variance_ratio_
            
            # 결과를 데이터프레임으로 변환
            importance_df = pd.DataFrame({
                'component': range(1, len(explained_variance_ratio) + 1),
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance_ratio': np.cumsum(explained_variance_ratio)
            })
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"특징 중요도 분석 중 오류 발생: {str(e)}")
            return pd.DataFrame()

    def save_analysis_results(self, results: Dict, filename: str):
        """
        분석 결과 저장
        
        Args:
            results (Dict): 저장할 분석 결과
            filename (str): 저장할 파일 이름
        """
        try:
            np.savez(
                f"data/analysis/{filename}.npz",
                **{k: v for k, v in results.items() if isinstance(v, np.ndarray)}
            )
            self.logger.info(f"분석 결과가 저장되었습니다: {filename}.npz")
        except Exception as e:
            self.logger.error(f"분석 결과 저장 중 오류 발생: {str(e)}")