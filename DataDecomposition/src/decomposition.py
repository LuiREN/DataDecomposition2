from sklearn.decomposition import TruncatedSVD, PCA, NMF, LatentDirichletAllocation
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def perform_decomposition(data, method='svd', n_components=2):
    if method == 'svd':
        decomposer = TruncatedSVD(n_components=n_components)
    elif method == 'pca':
        decomposer = PCA(n_components=n_components)
    elif method == 'nmf':
        decomposer = NMF(n_components=n_components)
    elif method == 'lda':
        decomposer = LatentDirichletAllocation(n_components=n_components)
    else:
        raise ValueError(f"Неизвестный метод декомпозиции: {method}")

    transformed_data = decomposer.fit_transform(data)
    return transformed_data

def plot_decomposition_results(self):
        figure = plt.figure(figsize=(5, 4))
        ax = figure.add_subplot(111)
        ax.scatter(data[:, 0], data[:, 1])
        ax.set_xlabel('Компонента 1')
        ax.set_ylabel('Компонента 2')
        ax.set_title('Результаты декомпозиции')
        return figure

class DecompositionDialog(QDialog):
    def __init__(self, data):
        super().__init__()
        self.setWindowTitle("Результаты декомпозиции")
        self.setGeometry(100, 100, 600, 400)
        
        layout = QVBoxLayout()
        
        figure = plt.figure(figsize=(5, 4))
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)
        ax.scatter(data[:, 0], data[:, 1])
        ax.set_xlabel('Компонента 1')
        ax.set_ylabel('Компонента 2')
        ax.set_title('Результаты декомпозиции')
        
        layout.addWidget(canvas)
        
        button = QPushButton("Закрыть")
        button.clicked.connect(self.close)
        layout.addWidget(button)
        
        self.setLayout(layout)
