import sys
from PyQt5.QtWidgets import QApplication
from csv_viewer import CsvViewer

if __name__ == '__main__': 
    app = QApplication(sys.argv)
    window = CsvViewer()
    window.show()
    sys.exit(app.exec_())
