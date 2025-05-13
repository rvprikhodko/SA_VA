import os
import pandas as pd
import ast
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
)

from ui.results_window import ResultsWindow


class LoadAnalysisWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Просмотр предыдущего анализа")
        self.setGeometry(300, 300, 400, 200)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Введите номер исследования:"))

        self.id_input = QLineEdit()
        layout.addWidget(self.id_input)

        self.load_btn = QPushButton("Загрузить анализ")
        self.load_btn.clicked.connect(self.load_analysis)
        layout.addWidget(self.load_btn)

        self.setLayout(layout)

    def load_analysis(self):
        analysis_id = self.id_input.text()
        if not analysis_id.isdigit():
            QMessageBox.warning(self, "Ошибка", "Номер исследования должен быть числом.")
            return

        csv_path = "data/analysis.csv"
        if not os.path.exists(csv_path):
            QMessageBox.critical(self, "Ошибка", "Файл analysis.csv не найден.")
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить CSV:\n{e}")
            return

        index = int(analysis_id)
        if index >= len(df):
            QMessageBox.information(self, "Не найдено", "Исследование с таким номером не найдено.")
            return

        try:
            row = df.iloc[index]

            patient_data = {
                "age": int(row["Возраст"]),
                "gender": row["Пол"],
                "FOV": ast.literal_eval(row["FOV"])
            }

            image_paths = ast.literal_eval(row["Путь к изображениям"])
            predicted_diameters = ast.literal_eval(row["Предсказанные диаметры артерий"])
            predicted_positions = ast.literal_eval(row["Предсказанные позиции артерий"])
            modified_diameters = ast.literal_eval(row["Измененные диаметры артерий"])
            modified_positions = ast.literal_eval(row["Измененные позиции артерий"])

            self.results_window = ResultsWindow(
                patient_data=patient_data,
                image_paths=image_paths,
                predicted_diameters=predicted_diameters,
                predicted_positions=predicted_positions
            )

            # Обновим поля изменённых данных
            self.results_window.modified_diameters = modified_diameters
            self.results_window.modified_positions = modified_positions
            self.results_window.update_display()
            self.results_window.show()
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка чтения", f"Ошибка при чтении анализа:\n{e}")
