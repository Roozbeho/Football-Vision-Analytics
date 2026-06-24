import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist
from collections import defaultdict
from typing import Dict, List
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import config

class TeamAssigner:
    def __init__(self, yolo_model):
        self.model = yolo_model
        # self.kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10, max_iter=300, random_state=42)
        # self.kmeans = GaussianMixture(
        #                 n_components=2,
        #                 covariance_type='full',
        #                 random_state=42
        #             )
        
        self.team_caches = {}

        self.player_colors = {}
        self.team_colors = {}
        
        self.cluster_centers = None
        self.is_fitted = False
        

    def _get_clustring(self, top_half):
        top_half_2d = top_half.reshape(-1,3)
        kmeans = KMeans(n_clusters=config.KMEANS_N_CLUSTERS,init='k-means++', n_init=1, random_state=0)
        kmeans.fit(top_half_2d)
        return kmeans
    
    def _get_player_color(self, frame, player):       
        bbox = player.bbox
        
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half_image = image[0: int(image.shape[0]//2),:]


        kmeans = self._get_clustring(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_clusters = [clustered_image[0,0],clustered_image[-1,0],clustered_image[0,-1],clustered_image[-1,-1]]
        background_cluster = max(corner_clusters, key=corner_clusters.count)
        player_cluster = 1 - background_cluster

        color = kmeans.cluster_centers_[player_cluster]
        
        return color
        
    def player_color_extraction(self, frame, players_data: List[Dict]):
        detect_player_colors = []
        for player in players_data:
            color = self._get_player_color(frame, player)
            detect_player_colors.append(color)
                
        kmeans = KMeans(n_clusters=2, init='k-means++',n_init=1)
        kmeans.fit(detect_player_colors)
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    
    
    def fit_model(self):
        ids, features = [], []
        
        for track_id, colors in self.player_colors.items():
            avg_color = np.mean(colors, axis=0)
            ids.append(track_id)
            features.append(avg_color)

        if len(ids) < 2:
            return None
        
        
        labels = self.kmeans.fit_predict(features)
        
        self.cluster_centers = self.kmeans.means_
        
        for track_id, label in zip(ids, labels):
            self.team_caches[track_id] = label
            
        self.is_fitted = True
            
    def get_players_team(self, frame, player):
        color = self._get_player_color(frame, player).reshape(1, -1)
        
        cluster_idx = self.kmeans.predict(color)[0] + 1
        
        return self.team_colors[cluster_idx], cluster_idx
