from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from ui.patient_info_window import PatientInfoWindow
from ui.load_analysis_window import LoadAnalysisWindow
from PyQt5.QtGui import QIcon
import os

class MainWindow(QWidget):
    def __init__(self, diameter_model, position_model):
        super().__init__()
        self.setWindowTitle("Анализ позвоночных артерий")
        self.setGeometry(100, 100, 300, 200)
        icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../icon/icon.ico"))
        self.setWindowIcon(QIcon(icon_path))
        self.diameter_model = diameter_model
        self.position_model = position_model

        layout = QVBoxLayout()

        self.analyze_button = QPushButton("Сделать анализ")
        self.analyze_button.clicked.connect(self.open_patient_info)
        layout.addWidget(self.analyze_button)

        self.view_analysis_button = QPushButton("Просмотреть предыдущий анализ")
        self.view_analysis_button.clicked.connect(self.open_analysis_view)
        layout.addWidget(self.view_analysis_button)

        self.setLayout(layout)

    def open_patient_info(self):
        self.patient_info_window = PatientInfoWindow(self.diameter_model, self.position_model)
        self.patient_info_window.show()

    def open_analysis_view(self):
        self.analysis_load_window = LoadAnalysisWindow()
        self.analysis_load_window.show()
