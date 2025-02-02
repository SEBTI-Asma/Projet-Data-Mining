import pandas as pd
import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def dbscan(data, eps, min_pts):
    """
    DBSCAN algorithm from scratch.
    :param data: 2D numpy array of data points.
    :param eps: Radius for neighborhood.
    :param min_pts: Minimum points required to form a dense region.
    :return: Cluster labels for each point (-1 for noise).
    """
    labels = [-1] * len(data)
    cluster_id = 0
    visited = [False] * len(data)

    def get_neighbors(point_idx):
        """Find all neighbors within eps radius."""
        neighbors = []
        for i in range(len(data)):
            if np.linalg.norm(data[point_idx] - data[i]) <= eps:  # Euclidean distance
                neighbors.append(i)
        return neighbors

    def expand_cluster(point_idx, neighbors):
        nonlocal cluster_id
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = get_neighbors(neighbor_idx)
                if len(new_neighbors) >= min_pts:
                    neighbors += new_neighbors
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            i += 1

    for point_idx in range(len(data)):
        if visited[point_idx]:
            continue
        visited[point_idx] = True
        neighbors = get_neighbors(point_idx)
        if len(neighbors) < min_pts:
            labels[point_idx] = -1
        else:
            cluster_id += 1
            expand_cluster(point_idx, neighbors)

    return labels


# Classe pour Clarans
class DbScanAlgorithm:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def apply_Without_PCA(self, eps : float, min_pts : float, mode):

        # Apply PCA to the dataset
        if mode == "With PCA":
            # Apply PCA to the dataset
            pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization

            data_pca = pca.fit_transform(self.data)  # 'data' is the original dataset
            
            start_time = time.time()

            cluster_labels = dbscan(data_pca, eps=0.05, min_pts=5)

            end_time = time.time()
            training_time = end_time - start_time
            print(f"Temps d'entraînement : {training_time:.4f} secondes") 

            d = data_pca

        elif mode == "Without PCA":
        
            # Select numerical features
            features = self.data.select_dtypes(include=[np.number]).values

            start_time = time.time()

            cluster_labels = dbscan(features, eps, min_pts)

            end_time = time.time()
            training_time = end_time - start_time
            print(f"Temps d'entraînement : {training_time:.4f} secondes") 

            d = features

        # Calculate results
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        num_noise = cluster_labels.count(-1)

        return num_clusters, num_noise, training_time, d, cluster_labels
    
    def display_clusters(self, d, cluster_labels):
        plt.scatter(d[:, 0], d[:, 1], c=cluster_labels, cmap='viridis')
        plt.title("DBSCAN Clusters")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label='Cluster ID')
        plt.show()

    def evaluate_dbscan(self, cluster_labels, data):
        # MSE Calculation
        mse_values = []
        for cluster_idx in set(cluster_labels):
            if cluster_idx == -1:  # Skip noise points
                continue
            
            cluster_points = data[np.array(cluster_labels) == cluster_idx]
            centroid = np.mean(cluster_points, axis=0)  # Calculate centroid

            # Compute squared distances and cluster MSE
            distances = np.linalg.norm(cluster_points - centroid, axis=1) ** 2
            mse_values.append(np.mean(distances))

        # Calculate metrics
            unique_labels = set(cluster_labels)
            if len(unique_labels) > 1 and -1 in unique_labels:
                unique_labels.remove(-1)  # Exclude noise for metrics

            wc_sse = sum(
                sum(np.linalg.norm(data[i] - np.mean(data[np.array(cluster_labels) == cluster], axis=0)) ** 2
                    for i in range(len(data)) if cluster_labels[i] == cluster)
                for cluster in unique_labels
            )

            silhouette = silhouette_score(data, cluster_labels) if len(unique_labels) > 1 else None

        return mse_values, wc_sse, silhouette


