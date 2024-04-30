from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

class AnalysisResultsDialog(QDialog):
    def __init__(self, results):
        super().__init__()
        self.setWindowTitle("Результаты анализов")
        self.setGeometry(100, 100, 600, 400)
        
        layout = QVBoxLayout()
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(results)
        layout.addWidget(text_edit)
        
        button = QPushButton("Закрыть")
        button.clicked.connect(self.close)
        layout.addWidget(button)
        
        self.setLayout(layout)
