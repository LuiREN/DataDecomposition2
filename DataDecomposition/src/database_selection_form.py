from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QFileDialog, QTextEdit

class DatabaseSelectionForm(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор базы данных")
        self.setGeometry(100, 100, 600, 400)
        
        layout = QVBoxLayout()
        
        self.select_button = QPushButton("Обзор")
        self.select_button.clicked.connect(self.select_database)
        layout.addWidget(self.select_button)
        
        self.database_preview = QTextEdit()
        self.database_preview.setReadOnly(True)
        layout.addWidget(self.database_preview)
        
        self.back_button = QPushButton("Назад")
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)
        
        self.setLayout(layout)
        
        self.selected_database = None
    
    def select_database(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл данных", "", "All Files (*)")
        if file_path:
            # Загрузка и отображение базы данных
            self.selected_database = file_path
            with open(file_path, 'r') as file:
                self.database_preview.setText(file.read())
    
    def go_back(self):
        self.accept()
