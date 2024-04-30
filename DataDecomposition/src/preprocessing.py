import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from multiprocessing import Pool


def load_data(file_path):
    try:
        # Загрузка данных из файла
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            data = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        elif file_path.endswith('.txt') or file_path.endswith('.tsv'):
            data = pd.read_csv(file_path, sep='\t')
        elif file_path.endswith('.pkl'):
            data = pd.read_pickle(file_path)
        else:
            raise ValueError("Неподдерживаемый формат файла")

        return data

    except Exception as e:
        raise ValueError(f"Ошибка при загрузке данных: {str(e)}")


def preprocess_data(data):
    try:
        # Проверка наличия признаков
        if data.shape[1] == 0:
            raise ValueError("Набор данных не содержит признаков (столбцов)")

        # Очистка данных от шумов и выбросов
        data = data.dropna(thresh=data.shape[1] * 0.7)  # Удаление строк с более чем 30% пропущенных значений

        # Удаление лишних пробелов в строковых данных
        data = data.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Создание экземпляра KNNImputer
        imputer = KNNImputer()

        # Обработка пропущенных значений с использованием KNN
        pool = Pool()
        numeric_columns = data.select_dtypes(include=[float, int]).columns
        data[numeric_columns] = pool.apply(imputer.fit_transform, args=(data[numeric_columns],))
        pool.close()
        pool.join()

        # Приведение числовых данных к диапазону [0, 1]
        scaler = MinMaxScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        # One-hot encoding для категориальных признаков
        categorical_columns = data.select_dtypes(include=['object']).columns
        encoded_data = pd.get_dummies(data[categorical_columns])
        data = pd.concat([data.drop(columns=categorical_columns), encoded_data], axis=1)

        return data

    except Exception as e:
        raise ValueError(f"Ошибка при предобработке данных: {str(e)}")


# Используйте эту функцию для вызова ваших функций параллельно
def apply_preprocessing(file_path):
    data = load_data(file_path)
    processed_data = parallelize_dataframe(data, preprocess_data)
    return processed_data

# Пример использования
if __name__ == "__main__":
    file_path = 'your_data.csv'  # Замените на путь к вашему файлу
    processed_data = apply_preprocessing(file_path)
    print(processed_data.head())
