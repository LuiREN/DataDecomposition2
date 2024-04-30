from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit

class DataPreviewForm(QDialog):
    def __init__(self, data):
        super().__init__()
        self.setWindowTitle("Предпросмотр данных")
        self.setGeometry(100, 100, 600, 400)
        
        layout = QVBoxLayout()
        
        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        self.data_preview.setText(data.head().to_string(index=False))
        layout.addWidget(self.data_preview)
        
        self.setLayout(layout)
