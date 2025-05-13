import os
import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt
from PIL import Image
import torchvision.transforms as transforms
import torch

# Подключим модельные классы
from models.diameter_model import VertebralArteryRegressor
from models.position_model import ArteriesLocalizationModel
from ui.results_window import ResultsWindow


class ImageUploadWindow(QWidget):
    def __init__(self, patient_data, diameter_model, position_model):
        super().__init__()
        self.setWindowTitle("Загрузка изображений")
        self.setGeometry(200, 200, 400, 200)
        self.patient_data = patient_data
        self.image_paths = []

        layout = QVBoxLayout()

        self.label = QLabel("Выберите изображения для анализа (.jpg, .png)")
        layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.upload_button = QPushButton("Загрузить изображения")
        self.upload_button.clicked.connect(self.upload_images)
        layout.addWidget(self.upload_button)

        self.setLayout(layout)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diameter_model = diameter_model
        self.position_model = position_model
        self.diameter_model.eval()
        self.position_model.eval()

    def upload_images(self):
        file_dialog = QFileDialog()
        paths, _ = file_dialog.getOpenFileNames(self, "Выберите изображения", "", "Images (*.jpg *.png)")

        if not paths:
            QMessageBox.information(self, "Информация", "Файлы не были выбраны.")
            return

        self.image_paths = paths
        self.analyze_images()

    def analyze_images(self):
        # Преобразования
        from utils.transforms import diameter_transform, position_transform

        diameters = []
        positions = []
        # self.diameter_model.load_state_dict(torch.load('weights/best_vertebral_artery_augmented_ants.pth', map_location=self.device))
        # self.diameter_model.eval()
        for path in self.image_paths:
            image = Image.open(path).convert("RGB")

            # Диаметры
            input_tensor = diameter_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_diameters = self.diameter_model(input_tensor).squeeze().tolist()
                diameters.append(tuple(pred_diameters))

            # Позиции
            input_tensor_pos = position_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_positions = self.position_model(input_tensor_pos).squeeze().tolist()
                # Получаем два кортежа (x1, y1), (x2, y2)
                pred_positions = [(pred_positions[0], pred_positions[1]), (pred_positions[2], pred_positions[3])]
                positions.append(pred_positions)

        self.results_window = ResultsWindow(
            patient_data=self.patient_data,
            image_paths=self.image_paths,
            predicted_diameters=diameters,
            predicted_positions=positions
        )
        self.results_window.show()
        self.close()
