import sys
import os
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'