from PyQt5.QtWidgets import QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QTextEdit, QComboBox, QMessageBox, QSpinBox, QCheckBox, QApplication, QTableWidget, QTableWidgetItem, QLineEdit, QPlainTextEdit, QTabWidget, QGroupBox, QScrollArea,QAction, QDoubleSpinBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread, pyqtSignal, QThreadPool
from src.preprocessing import preprocess_data, load_data
from src.feature_importance import get_feature_importance, identify_uninformative_features
from src.semantic_db import create_ontology,create_semantic_db
from src.decomposition import perform_decomposition, DecompositionDialog
from src.clustering import perform_clustering
from src.evaluation import calculate_metrics, ComparisonDialog
from src.feature_selection import anova_feature_selection, mutual_information_feature_selection
from src.analysis_results import AnalysisResultsDialog
from gui.loading_dialog import LoadingDialog
import pandas as pd
import pickle
import sys


class DataLoadingThread(QThread):
    data_loaded = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        

    def run(self):
        try:
            data = load_data(self.file_path)
            self.data_loaded.emit(data)
        except FileNotFoundError:
            self.error_occurred.emit("Файл не найден")
        except Exception as e:
            self.error_occurred.emit(f"Ошибка при загрузке данных: {str(e)}")

class DataProcessingThread(QThread):
    data_processed = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)

    def __init__(self, data):
        super().__init__()
        self.data = data
       
    def run(self):
        try:
            pool = Pool()
            result = pool.apply_async(preprocess_data, (self.data))
            data = result.get()
            self.data_processed.emit(data)
        except ValueError as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            self.error_occurred.emit(f"Ошибка при предобработке данных: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Decomposition")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.tab_widget = QTabWidget(self)
        self.main_layout.addWidget(self.tab_widget)

        self.menu_bar = self.menuBar()
        self.project_menu = self.menu_bar.addMenu("Проект")

        self.save_project_action = QAction("Сохранить проект", self)
        self.save_project_action.triggered.connect(self.save_project)
        self.project_menu.addAction(self.save_project_action)

        self.load_project_action = QAction("Загрузить проект", self)
        self.load_project_action.triggered.connect(self.load_project)
        self.project_menu.addAction(self.load_project_action)

        self.data_tab = QWidget()
        self.data_layout = QVBoxLayout(self.data_tab)
        self.tab_widget.addTab(self.data_tab, "Данные")

        self.select_data_button = QPushButton("Выбрать файл данных")
        self.select_data_button.clicked.connect(self.select_data)
        self.data_layout.addWidget(self.select_data_button)

        self.preprocess_data_button = QPushButton("Предобработка данных")
        self.preprocess_data_button.clicked.connect(self.preprocess_data)
        self.data_layout.addWidget(self.preprocess_data_button)

        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        self.data_layout.addWidget(self.data_preview)

        self.analysis_tab = QWidget()
        self.analysis_layout = QVBoxLayout(self.analysis_tab)
        self.tab_widget.addTab(self.analysis_tab, "Анализ")

        self.feature_selection_group = QGroupBox("Выбор признаков")
        self.feature_selection_layout = QVBoxLayout(self.feature_selection_group)
        self.analysis_layout.addWidget(self.feature_selection_group)

        self.discretize_button = QPushButton("Преобразовать целевую переменную")
        self.discretize_button.clicked.connect(self.discretize_target)
        self.discretize_button.setVisible(False)
        self.feature_selection_layout.addWidget(self.discretize_button)

        self.feature_selection_combo = QComboBox()
        self.feature_selection_combo.addItems(["ANOVA", "Mutual Information"])
        self.feature_selection_layout.addWidget(self.feature_selection_combo)

        self.feature_selection_combo.currentTextChanged.connect(self.update_discretize_button)

        self.k_features_spinbox = QSpinBox()
        self.k_features_spinbox.setMinimum(1)
        self.k_features_spinbox.setMaximum(100)
        self.k_features_spinbox.setValue(10)
        self.feature_selection_layout.addWidget(self.k_features_spinbox)

        self.feature_selection_button = QPushButton("Выбрать признаки")
        self.feature_selection_button.clicked.connect(self.perform_feature_selection)
        self.feature_selection_layout.addWidget(self.feature_selection_button)

        self.decomposition_group = QGroupBox("Декомпозиция")
        self.decomposition_layout = QVBoxLayout(self.decomposition_group)
        self.analysis_layout.addWidget(self.decomposition_group)

        self.decomposition_combo = QComboBox()
        self.decomposition_combo.addItems(["SVD", "PCA", "NMF", "LDA"])
        self.decomposition_layout.addWidget(self.decomposition_combo)

        self.n_components_spinbox = QSpinBox()
        self.n_components_spinbox.setMinimum(1)
        self.n_components_spinbox.setMaximum(100)
        self.n_components_spinbox.setValue(2)
        self.decomposition_layout.addWidget(self.n_components_spinbox)

        self.decomposition_button = QPushButton("Выполнить декомпозицию")
        self.decomposition_button.clicked.connect(self.perform_decomposition)
        self.decomposition_layout.addWidget(self.decomposition_button)

        self.clustering_group = QGroupBox("Кластеризация")
        self.clustering_layout = QVBoxLayout(self.clustering_group)
        self.analysis_layout.addWidget(self.clustering_group)

        self.clustering_combo = QComboBox()
        self.clustering_combo.addItems(["K-means", "Hierarchical", "DBSCAN"])
        self.clustering_layout.addWidget(self.clustering_combo)

        self.clustering_button = QPushButton("Выполнить кластеризацию")
        self.clustering_button.clicked.connect(self.perform_clustering)
        self.clustering_layout.addWidget(self.clustering_button)

        self.comparison_group = QGroupBox("Сравнение алгоритмов")
        self.comparison_layout = QVBoxLayout(self.comparison_group)
        self.analysis_layout.addWidget(self.comparison_group)

        self.comparison_button = QPushButton("Сравнить алгоритмы")
        self.comparison_button.clicked.connect(self.compare_algorithms)
        self.comparison_layout.addWidget(self.comparison_button)

        self.semantic_db_tab = QWidget()
        self.semantic_db_layout = QVBoxLayout(self.semantic_db_tab)
        self.tab_widget.addTab(self.semantic_db_tab, "Семантическая БД")

        self.semantic_db_button = QPushButton("Преобразовать в семантическую БД")
        self.semantic_db_button.clicked.connect(self.create_semantic_db)
        self.semantic_db_layout.addWidget(self.semantic_db_button)

        self.semantic_db_result = QTextEdit()
        self.semantic_db_result.setReadOnly(True)
        self.semantic_db_layout.addWidget(self.semantic_db_result)
        
        self.group_layout = None
        self.feature_importance_tab = QWidget()
        self.feature_importance_layout = QVBoxLayout(self.feature_importance_tab)
        self.tab_widget.addTab(self.feature_importance_tab, "Важность признаков")

        self.feature_importance_group = QGroupBox("Параметры")
        self.feature_importance_layout.addWidget(self.feature_importance_group)
        self.group_layout = QHBoxLayout(self.feature_importance_group)

        self.target_column_combo = QComboBox()
        self.group_layout.addWidget(QLabel("Целевая переменная:"))
        self.group_layout.addWidget(self.target_column_combo)

        self.correlation_threshold_spin = QDoubleSpinBox()
        self.correlation_threshold_spin.setMinimum(0.0)
        self.correlation_threshold_spin.setMaximum(1.0)
        self.correlation_threshold_spin.setValue(0.1)
        self.correlation_threshold_spin.setSingleStep(0.1)
        self.group_layout.addWidget(QLabel("Порог корреляции:"))
        self.group_layout.addWidget(self.correlation_threshold_spin)

        self.categorical_threshold_spin = QSpinBox()
        self.categorical_threshold_spin.setMinimum(2)
        self.categorical_threshold_spin.setMaximum(100)
        self.categorical_threshold_spin.setValue(5)
        self.group_layout.addWidget(QLabel("Порог категориальных:"))
        self.group_layout.addWidget(self.categorical_threshold_spin)

        self.identify_features_button = QPushButton("Определить малоинформативные признаки")
        self.identify_features_button.clicked.connect(self.identify_uninformative_features)
        self.feature_importance_layout.addWidget(self.identify_features_button)

        self.feature_importance_result = QTextEdit()
        self.feature_importance_result.setReadOnly(True)
        self.feature_importance_layout.addWidget(self.feature_importance_result)

        self.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QTabWidget::pane {
            border: none;
        }
        QTabBar::tab {
            background-color: #e0e0e0;
            padding: 10px;
        }
        QTabBar::tab:selected {
            background-color: #c0c0c0;
        }
        QGroupBox {
            background-color: #f5f5f5;
            border: 1px solid #c0c0c0;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
        }
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QLabel {
            font-size: 14px;
        }
        QComboBox, QSpinBox {
            padding: 5px;
            font-size: 14px;
        }
    """)

    def disable_buttons(self):
        self.select_data_button.setEnabled(False)
        self.feature_selection_button.setEnabled(False)
        self.decomposition_button.setEnabled(False)
        self.clustering_button.setEnabled(False)
        self.comparison_button.setEnabled(False)
        self.save_project_action.setEnabled(False)
        self.load_project_action.setEnabled(False)

    def enable_buttons(self):
        self.select_data_button.setEnabled(True)
        self.feature_selection_button.setEnabled(True)
        self.decomposition_button.setEnabled(True)
        self.clustering_button.setEnabled(True)
        self.comparison_button.setEnabled(True)
        self.save_project_action.setEnabled(True)
        self.load_project_action.setEnabled(True)

    def select_data(self):
        self.disable_buttons()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл данных", "", "All Files (*)")
        if file_path:
            self.loading_dialog = LoadingDialog(self)
            self.data_loading_thread = DataLoadingThread(file_path)
            self.data_loading_thread.data_loaded.connect(self.on_data_loaded)
            self.data_loading_thread.error_occurred.connect(self.on_error_occurred)
            self.data_loading_thread.finished.connect(self.on_data_loading_finished)
            self.data_loading_thread.start()
            self.loading_dialog.exec_()
        self.enable_buttons()

    def on_data_loaded(self, data):
        self.data = data
        self.data_preview.setText(data.to_string(index=False))
        QMessageBox.information(self, "Успех", "Данные загружены")
        self.loading_dialog.close()
        # Заполнить combobox для выбора целевой переменной
        self.target_column_combo.clear()
        self.target_column_combo.addItems(self.data.columns.tolist())
        

    def on_data_loading_finished(self):
        self.enable_buttons()
    
    def on_error_occurred(self, error_message):
        QMessageBox.critical(self, "Ошибка", error_message)
        self.loading_dialog.close()
        self.enable_buttons()

    def preprocess_data(self):
        if hasattr(self, 'data'):
            try:
                self.data = preprocess_data(self.data)
                QMessageBox.information(self, "Успех", "Данные предобработаны")
                self.data_preview.setText(self.data.head().to_string(index=False))
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
        else:
            QMessageBox.warning(self, "Предупреждение", "Данные не загружены. Пожалуйста, загрузите данные перед предобработкой.")

    def perform_feature_selection(self):
        if hasattr(self, 'data'):
            self.disable_buttons()
            try:
                method = self.feature_selection_combo.currentText().lower()
                k = self.k_features_spinbox.value()
                target = self.data.iloc[:, -1]  # Считаем, что целевая переменная находится в последнем столбце
                
                if method == "anova":
                    self.data = anova_feature_selection(self.data.iloc[:, :-1], target, k=k)
                elif method == "mutual information":
                    self.data = mutual_information_feature_selection(self.data.iloc[:, :-1], target, k=k)
                
                QMessageBox.information(self, "Успех", "Признаки выбраны")
                self.show_analysis_results(f"Метод выбора признаков: {method}\nКоличество признаков: {k}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
            self.enable_buttons()
        else:
            QMessageBox.warning(self, "Предупреждение", "Данные не загружены. Пожалуйста, загрузите и предобработайте данные перед выбором признаков.")

    def perform_decomposition(self):
        if hasattr(self, 'data'):
            self.disable_buttons()
            try:
                method = self.decomposition_combo.currentText().lower()
                n_components = self.n_components_spinbox.value()
                n_features = self.data.shape[1]
                if n_components > n_features:
                    raise ValueError(f"Количество компонент ({n_components}) должно быть меньше или равно количеству признаков ({n_features})")
                self.transformed_data = perform_decomposition(self.data, method=method, n_components=n_components)
                QMessageBox.information(self, "Успех", "Данные разложены с помощью выбранного метода декомпозиции")
                
                dialog = DecompositionDialog(self.transformed_data)
                dialog.exec_()
                
                self.show_analysis_results(f"Метод декомпозиции: {method}\nКоличество компонент: {n_components}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
            self.enable_buttons()
        else:
            QMessageBox.warning(self, "Предупреждение", "Данные не загружены. Пожалуйста, загрузите и предобработайте данные перед выполнением декомпозиции.")

    def perform_clustering(self):
        if hasattr(self, 'data'):
            self.disable_buttons()
            try:
                method = self.clustering_combo.currentText().lower()
                labels, metrics = perform_clustering(self.data, method=method)
                QMessageBox.information(self, "Успех", "Данные кластеризованы")
                
                self.show_analysis_results(f"Метод кластеризации: {method}\n\nМетки кластеров:\n{pd.Series(labels).to_string(index=False)}\n\nМетрики качества:\n{pd.Series(metrics).to_string()}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
            self.enable_buttons()
        else:
            QMessageBox.warning(self, "Предупреждение", "Данные не загружены. Пожалуйста, загрузите и предобработайте данные перед выполнением кластеризации.")

    def compare_algorithms(self):
        if hasattr(self, 'data'):
            self.disable_buttons()
            try:
                algorithms = ['svd', 'pca', 'nmf', 'lda']
                n_components = self.n_components_spinbox.value()

                results = {}
                metrics = {}
                for algorithm in algorithms:
                    transformed_data = perform_decomposition(self.data, method=algorithm, n_components=n_components)
                    results[algorithm] = transformed_data
                    metrics[algorithm] = calculate_metrics(self.data, transformed_data)
            
                dialog = ComparisonDialog(results, metrics)
                dialog.exec_()
            
                self.show_analysis_results(f"Сравнение алгоритмов декомпозиции:\n\n{pd.DataFrame.from_dict(metrics, orient='index').to_string()}")
            
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
            self.enable_buttons()
        else:
            QMessageBox.warning(self, "Предупреждение", "Данные не загружены. Пожалуйста, загрузите и предобработайте данные перед сравнением алгоритмов.")
    
    def show_analysis_results(self, results):
        dialog = AnalysisResultsDialog(results)
        dialog.exec_()

    def save_project(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "", "Pickle Files (*.pkl)")
            if file_path:
                project_data = {
                    'data': self.data if hasattr(self, 'data') else None,
                    'ontology': self.ontology if hasattr(self, 'ontology') else None,
                    'transformed_data': self.transformed_data if hasattr(self, 'transformed_data') else None
                }
                with open(file_path, 'wb') as file:
                    pickle.dump(project_data, file)
                QMessageBox.information(self, "Успех", "Проект сохранен")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def load_project(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить проект", "", "Pickle Files (*.pkl)")
            if file_path:
                with open(file_path, 'rb') as file:
                    project_data = pickle.load(file)
                self.data = project_data.get('data')
                self.ontology = project_data.get('ontology')
                self.transformed_data = project_data.get('transformed_data')
                QMessageBox.information(self, "Успех", "Проект загружен")
                if self.data is not None:
                    self.data_preview.setText(self.data.head().to_string(index=False))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def update_discretize_button(self, text):
        if text == "Mutual Information":
            self.discretize_button.setVisible(True)
        else:
            self.discretize_button.setVisible(False)

    def discretize_target(self):
        if hasattr(self, 'data'):
            try:
                target = self.data.iloc[:, -1]
                discretized_target = pd.qcut(target, q=5, labels=False)
                self.data.iloc[:, -1] = discretized_target
                QMessageBox.information(self, "Успех", "Целевая переменная преобразована в дискретную форму")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
        else:
            QMessageBox.warning(self, "Предупреждение", "Данные не загружены. Пожалуйста, загрузите данные перед преобразованием целевой переменной.")

    def create_semantic_db(self):
        if hasattr(self, 'data'):
            try:
                self.ontology = create_ontology(self.data)
                semantic_db = self.ontology.serialize(format='turtle')
                self.semantic_db_result.setText(semantic_db)
                QMessageBox.information(self, "Успех", "Данные преобразованы в семантическую базу")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
        else:
            QMessageBox.warning(self, "Предупреждение", "Данные не загружены. Пожалуйста, загрузите и предобработайте данные перед преобразованием в семантическую базу данных.")

    def identify_uninformative_features(self):
        if hasattr(self, 'data') and self.ontology is not None:
            target_column = self.data.columns[self.target_column_combo.currentIndex()]
            correlation_threshold = self.correlation_threshold_spin.value()
            categorical_threshold = self.categorical_threshold_spin.value()

            feature_importance = get_feature_importance(self.data, self.ontology, target_column)
            uninformative_features = identify_uninformative_features(feature_importance, correlation_threshold, categorical_threshold)

            result_text = "Малоинформативные признаки:\n\n"
            result_text += "\n".join(uninformative_features)

            self.feature_importance_result.setText(result_text)
        else:
            QMessageBox.warning(self, "Предупреждение", "Данные или онтология не загружены. Пожалуйста, загрузите данные и создайте семантическую базу данных перед определением малоинформативных признаков.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
