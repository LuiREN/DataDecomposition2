from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_metrics(original_data, transformed_data):
    metrics = {}
    n_components = transformed_data.shape[1]

    if n_components == 1:
        labels = transformed_data.flatten()
    else:
        kmeans = KMeans(n_clusters=min(original_data.shape[0], 8), random_state=42)
        labels = kmeans.fit_predict(transformed_data)

    n_samples = original_data.shape[0]
    n_clusters = len(np.unique(labels))

    if n_clusters < 2 or n_clusters >= n_samples:
        return metrics

    metrics['davies_bouldin'] = davies_bouldin_score(original_data, labels)

    if 2 <= n_clusters <= n_samples - 1:
        metrics['silhouette'] = silhouette_score(original_data, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(original_data, labels)

    return metricss

class ComparisonDialog(QDialog):
    def __init__(self, results, metrics):
        super().__init__()
        self.setWindowTitle("Сравнение алгоритмов")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        
        figure = plt.figure(figsize=(7, 5))
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)
        for algorithm, transformed_data in results.items():
            ax.scatter(transformed_data[:, 0], transformed_data[:, 1], label=algorithm)
        ax.set_xlabel('Компонента 1')
        ax.set_ylabel('Компонента 2')
        ax.set_title('Сравнение алгоритмов декомпозиции')
        ax.legend()
        
        layout.addWidget(canvas)
        
        metrics_text = "Метрики качества:\n"
        for algorithm, metric_values in metrics.items():
            metrics_text += f"Алгоритм: {algorithm}\n"
            metrics_text += f"{pd.Series(metric_values).to_string()}\n\n"
        
        metrics_label = QLabel(metrics_text)
        layout.addWidget(metrics_label)
        
        button = QPushButton("Закрыть")
        button.clicked.connect(self.close)
        layout.addWidget(button)
        
        self.setLayout(layout)
