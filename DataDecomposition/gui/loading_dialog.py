from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar

class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Загрузка данных")
        self.setFixedSize(300, 100)

        layout = QVBoxLayout()
        self.label = QLabel("Загрузка данных...")
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def set_label_text(self, text):
        self.label.setText(text)
