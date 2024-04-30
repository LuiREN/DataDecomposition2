from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def perform_svd(data, n_components=2):
    # Создание объекта SVD
    svd = TruncatedSVD(n_components=n_components, algorithm='randomized')

    # Выполнение SVD
    transformed_data = svd.fit_transform(data)

    # Нормализация данных после SVD
    normalized_data = normalize(transformed_data)

    return normalized_data