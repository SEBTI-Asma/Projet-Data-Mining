import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class CLARANS:
    def __init__(self, num_clusters, max_neighbors, num_local, distance_metric="euclidean"):
        """
        Initialize the CLARANS algorithm.

        Parameters:
            num_clusters: Number of clusters (k).
            max_neighbors: Maximum number of neighbors to explore.
            num_local: Number of local search trials.
            distance_metric: Distance metric ("euclidean" or "manhattan").
        """
        self.num_clusters = num_clusters
        self.max_neighbors = max_neighbors
        self.num_local = num_local
        self.distance_metric = distance_metric

    def compute_distance(self, point1, point2):
        """Calculate distance between two points."""
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(point1 - point2))
        else:
            raise ValueError("Unsupported distance metric.")

    def assign_to_clusters(self, data, medoids):
        """Assign each data point to the nearest medoid."""
        clusters = {i: [] for i in range(len(medoids))}
        cost = 0
        
        for point in data:
            distances = [self.compute_distance(point, medoid) for medoid in medoids]
            nearest_medoid = np.argmin(distances)
            clusters[nearest_medoid].append(point)
            cost += distances[nearest_medoid]

        return clusters, cost

    def find_best_medoid(self, cluster_points, current_medoid):
        """Find the best medoid for a given cluster."""
        min_cost = float("inf")
        best_medoid = current_medoid

        for candidate_medoid in cluster_points:
            cost = sum(self.compute_distance(candidate_medoid, point) for point in cluster_points)
            if cost < min_cost:
                min_cost = cost
                best_medoid = candidate_medoid

        return best_medoid

    def fit(self, data):
        """
        Run the CLARANS algorithm on the data.

        Parameters:
            data: Data to be clustered, in numpy.ndarray format.

        Returns:
            A tuple containing the final clusters and the final medoids.
        """
        best_medoids = None
        best_cost = float("inf")
        best_clusters = None

        for _ in range(self.num_local):
            # Randomly initialize the medoids
            medoids_indices = random.sample(range(len(data)), self.num_clusters)
            medoids = [data[idx] for idx in medoids_indices]

            improved = True
            while improved:
                improved = False
                current_clusters, current_cost = self.assign_to_clusters(data, medoids)
                
                # Explore neighbors
                for _ in range(self.max_neighbors):
                    # Randomly select a medoid and an alternative
                    medoid_idx = random.choice(range(len(medoids)))
                    alternative_idx = random.choice([i for i in range(len(data)) if i not in medoids_indices])
                    alternative_medoid = data[alternative_idx]

                    # Create a new set of medoids
                    new_medoids = medoids[:]
                    new_medoids[medoid_idx] = alternative_medoid

                    # Compute the cost for the new medoids
                    _, new_cost = self.assign_to_clusters(data, new_medoids)

                    if new_cost < current_cost:
                        medoids = new_medoids
                        medoids_indices[medoid_idx] = alternative_idx
                        current_cost = new_cost
                        improved = True
                        break

            # Update if a better local solution is found
            if current_cost < best_cost:
                best_medoids = medoids
                best_cost = current_cost
                best_clusters = current_clusters

        return best_clusters, best_medoids

# Classe pour Clarans
class ClaransAlgorithm:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def apply(self, num_clusters : int, max_neighbors : int, num_local : int, mode):
       
        # We will exclude 'latitude', 'longitude', 'geometry' and other non-numeric columns.
        columns_to_use = [col for col in self.data.columns if self.data[col].dtype in ['float64', 'int64']]
        data_for_clustering = self.data[columns_to_use].values

        # Apply the CLARANS algorithm
        clarans = CLARANS(num_clusters=num_clusters, max_neighbors=max_neighbors, num_local=num_local, distance_metric="euclidean")
        
        # Apply PCA to the dataset
        if mode == "With PCA":
            # Apply PCA to reduce dimensionality
            pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
            data_pca = pca.fit_transform(data_for_clustering)

            # Mesurer le temps d'exécution pour l'entraînement
            start_time = time.time()

            clusters, medoids = clarans.fit(data_pca)

            end_time = time.time()
            training_time = end_time - start_time
            print(f"Temps d'entraînement : {training_time:.4f} secondes") 

            d = data_pca
            
        elif mode == "Without PCA":
            # Mesurer le temps d'exécution pour l'entraînement
            start_time = time.time()

            clusters, medoids = clarans.fit(data_for_clustering)

            end_time = time.time()
            training_time = end_time - start_time
            print(f"Temps d'entraînement : {training_time:.4f} secondes") 
            d = data_for_clustering

        return clusters, medoids, training_time, d
    
    def display_clusters(self, clusters, medoids, data_for_clustering, data):
        # Add cluster labels to the original data for visualization
        data['cluster'] = -1  # Default label
        for cluster_idx, points in clusters.items():
            for point in points:
                idx = np.where((data_for_clustering == point).all(axis=1))[0][0]
                data.at[idx, 'cluster'] = cluster_idx

        # Extract latitude and longitude for visualization
        latitude = data['latitude']
        longitude = data['longitude']
        cluster_labels = data['cluster']

        # Plot the results
        plt.figure(figsize=(10, 6))
        unique_labels = set(cluster_labels)
        for cluster_id in unique_labels:
            cluster_points = data[data['cluster'] == cluster_id]
            if cluster_id == -1:  # Noise points
                plt.scatter(cluster_points['longitude'], cluster_points['latitude'],
                            color='k', label='Noise', alpha=0.5)
            else:
                plt.scatter(cluster_points['longitude'], cluster_points['latitude'],
                            label=f'Cluster {cluster_id}', alpha=0.6)

        # Highlight the medoids
        medoids_array = np.array(medoids)
        plt.scatter(medoids_array[:, 1], medoids_array[:, 0], color='red', marker='x', s=200, label='Medoids')

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("CLARANS Clusters (Latitude vs Longitude)")
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))  # Adjust legend position
        plt.grid()
        plt.show()

    def display_clusters_PCA(self, clusters, medoids, data_pca, data):
        # Prepare data for plotting
        data['cluster'] = -1
        for cluster_idx, points in clusters.items():
            for point in points:
                idx = np.where((data_pca == point).all(axis=1))[0][0]
                data.loc[idx, 'cluster'] = cluster_idx

        # Plot the results
        plt.figure(figsize=(10, 6))
        unique_labels = set(data['cluster'])
        for cluster_id in unique_labels:
            cluster_points = np.array([data_pca[i] for i in range(len(data_pca)) if data['cluster'].iloc[i] == cluster_id])
            if cluster_id == -1:  # Noise points
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color='k', label='Noise', alpha=0.5)
            else:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}', alpha=0.6)

        # Plot medoids
        medoids = np.array(medoids)
        plt.scatter(medoids[:, 0], medoids[:, 1], color="red", marker="x", s=200, label="Medoids")

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("CLARANS Clusters (PCA Components)")
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

    def evaluate_clarans(self, clusters, medoids, data_for_clustering):

        # Calcul du MSE pour chaque cluster
        mse_values = []
        for cluster_idx, cluster_points in clusters.items():
            cluster_points = np.array(cluster_points, dtype=int)  # Convert to integer indices
            cluster_data = data_for_clustering[cluster_points]  # Access the cluster's points
            medoid = medoids[cluster_idx]  # Get the medoid for the current cluster

            # Calculer la distance entre chaque point et le médoïde
            distances = np.linalg.norm(cluster_data - medoid, axis=1)
            mse = np.mean(distances ** 2)  # Erreur quadratique moyenne pour le cluster
            mse_values.append(mse)

        # Calculate metrics
        cluster_labels = np.zeros(len(data_for_clustering), dtype=int)
        for cluster_idx, points in clusters.items():
            for point in points:
                idx = np.where((data_for_clustering == point).all(axis=1))[0][0]
                cluster_labels[idx] = cluster_idx

        wc_sse = sum(
                    sum(np.linalg.norm(point - medoids[cluster_idx]) ** 2 for point in points)
                    for cluster_idx, points in clusters.items()
                )
        
        silhouette = silhouette_score(data_for_clustering, cluster_labels) if len(np.unique(cluster_labels)) > 1 else None

        return mse_values, wc_sse, silhouette
        
    
    

