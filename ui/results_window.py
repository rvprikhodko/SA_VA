from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QFileDialog
)
from PyQt5.QtCore import Qt
import math
import os
import csv
import pandas as pd

from ui.diameter_edit_window import DiameterEditWindow
from ui.positions_view_window import PositionsViewWindow


class ResultsWindow(QWidget):
    def __init__(self, patient_data, image_paths, predicted_diameters, predicted_positions):
        super().__init__()
        self.setWindowTitle("Результаты анализа")
        self.setGeometry(250, 250, 500, 400)

        self.patient_data = patient_data
        self.image_paths = image_paths
        self.predicted_diameters = predicted_diameters  # [(L, R), ...]
        self.modified_diameters = predicted_diameters.copy()
        self.predicted_positions = predicted_positions  # [[(x1, y1), (x2, y2)], ...]
        self.modified_positions = predicted_positions.copy()

        self.layout = QVBoxLayout()

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)

        self.btn_edit_diameter = QPushButton("Изменить диаметры позвоночных артерий")
        self.btn_edit_diameter.clicked.connect(self.edit_diameters)
        self.layout.addWidget(self.btn_edit_diameter)

        self.btn_view_positions = QPushButton("Просмотреть позиции артерий")
        self.btn_view_positions.clicked.connect(self.view_positions)
        self.layout.addWidget(self.btn_view_positions)

        self.save_button = QPushButton("Сохранить анализ")
        self.save_button.clicked.connect(self.save_analysis)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)
        self.update_display()

    def update_display(self):
        self.result_text.clear()

        def mm_area(d_mm):
            return round(math.pi * (d_mm / 2) ** 2, 2)

        # Попросим пользователя ввести FOV (размер изображения в см)
        fov_x, fov_y = self.patient_data['FOV'][0], self.patient_data['FOV'][1]  # временно зашито
        img_w, img_h = 512, 512  # placeholder
        px_to_mm_x = (fov_x * 10) / img_w
        px_to_mm_y = (fov_y * 10) / img_h
        px_to_mm = (px_to_mm_x + px_to_mm_y) / 2  # усредним

        diam_mm = [(round(l * px_to_mm_x, 2), round(r * px_to_mm_x, 2)) for l, r in self.modified_diameters]
        areas = [(mm_area(l), mm_area(r)) for l, r in diam_mm]

        lines = ["Диаметры (мм) и площади (мм²):\n"]
        for i, (d, a) in enumerate(zip(diam_mm, areas)):
            lines.append(f"  Изображение {i + 1}: L = {d[0]} мм / {a[0]} мм², R = {d[1]} мм / {a[1]} мм²")

        # Статистика
        l_diameters = [d[0] for d in diam_mm]
        r_diameters = [d[1] for d in diam_mm]
        lines.append("\nСтатистика:")
        lines.append(f"  Левый сосуд: min={min(l_diameters):.2f}, max={max(l_diameters):.2f}, mean={sum(l_diameters)/len(l_diameters):.2f}")
        lines.append(f"  Правый сосуд: min={min(r_diameters):.2f}, max={max(r_diameters):.2f}, mean={sum(r_diameters)/len(r_diameters):.2f}")

        self.result_text.setText("\n".join(lines))

    def edit_diameters(self):
        fov_x, fov_y = self.patient_data['FOV']
        self.edit_window = DiameterEditWindow(
            self.modified_diameters,
            self.update_display,
            fov_x=fov_x,
            fov_y=fov_y,
            img_width=512,
            img_height=512,
            image_paths=self.image_paths
        )
        self.edit_window.show()

    def view_positions(self):
        def update_positions_callback(new_positions):
            self.modified_positions = new_positions
            print("Позиции обновлены!")

        self.positions_window = PositionsViewWindow(
            self.image_paths,
            self.modified_positions,
            self.modified_diameters,
            update_callback=update_positions_callback
        )
        self.positions_window.show()

    def save_analysis(self):

        file_path = "data/analysis.csv"
        os.makedirs("data", exist_ok=True)

        def format_array(arr):
            return str(arr)

        row = {
            "Возраст": self.patient_data["age"],
            "Пол": self.patient_data["gender"],
            "FOV": self.patient_data['FOV'],
            "Путь к изображениям": format_array(self.image_paths),
            "Предсказанные диаметры артерий": format_array(self.predicted_diameters),
            "Предсказанные позиции артерий": format_array(self.predicted_positions),
            "Измененные диаметры артерий": format_array(self.modified_diameters),
            "Измененные позиции артерий": format_array(self.modified_positions),
        }

        # Загружаем или создаём датафрейм
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
        else:
            df = pd.DataFrame()

        # Добавляем новую строку
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # Сохраняем обратно
        df.to_csv(file_path, index=False, encoding='utf-8')

        # Показываем ID (индекс последней добавленной строки)
        new_id = len(df) - 1
        self.result_text.append(f"\nАнализ сохранён с ID: {new_id}")
