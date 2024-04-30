from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton,QScrollArea
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def perform_clustering(data, method='kmeans', **kwargs):
    if method == 'kmeans':
        n_clusters = kwargs.get('n_clusters', 3)
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(data)
    elif method == 'hierarchical':
        n_clusters = kwargs.get('n_clusters', 3)
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(data)
    else:
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
    
    try:
        labels = labels.flatten()

        # Вычисление метрик качества кластеризации
        if method != 'dbscan':
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            n_samples = data.shape[0]
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if 2 <= n_clusters < n_samples:
                silhouette = silhouette_score(data, labels)
                calinski_harabasz = calinski_harabasz_score(data, labels)
                davies_bouldin = davies_bouldin_score(data, labels)
                metrics = {
                    'Silhouette Score': silhouette,
                    'Calinski-Harabasz Index': calinski_harabasz,
                    'Davies-Bouldin Index': davies_bouldin
                }
            else:
                metrics = {}
        else:
            metrics = {}

    except ValueError as e:
        if "Number of labels is 1" in str(e):
            raise ValueError("Не удалось разделить данные на кластеры. Убедитесь, что данные содержат достаточно различий.")
        else:
            raise e

    return labels, metrics

