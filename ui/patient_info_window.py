from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QComboBox
)
from ui.image_upload_window import ImageUploadWindow
from PyQt5.QtGui import QDoubleValidator, QIntValidator

class PatientInfoWindow(QWidget):
    def __init__(self, diameter_model, position_model):
        super().__init__()
        self.setWindowTitle("Информация о пациенте")
        self.setGeometry(150, 150, 500, 200)
        self.diameter_model = diameter_model
        self.position_model = position_model

        layout = QVBoxLayout()

        # Пол (QComboBox вместо QLineEdit)
        self.gender_label = QLabel("Пол:")
        self.gender_input = QComboBox()
        self.gender_input.addItems(["М", "Ж"])
        layout.addWidget(self.gender_label)
        layout.addWidget(self.gender_input)

        # Возраст
        self.age_label = QLabel("Возраст (0 или больше):")
        self.age_input = QLineEdit()
        self.age_input.setValidator(QIntValidator(0, 150))  # Ограничим возраст разумными числами, например от 0 до 150
        layout.addWidget(self.age_label)
        layout.addWidget(self.age_input)

        # X
        self.X_label = QLabel("Размеры оси X в см:")
        self.X_input = QLineEdit()
        self.X_input.setValidator(
            QDoubleValidator(0.01, 1000.0, 2))  # Допустимый диапазон: от 0.01 до 1000.00, 2 знака после запятой
        layout.addWidget(self.X_label)
        layout.addWidget(self.X_input)

        # Y
        self.Y_label = QLabel("Размеры оси Y в см:")
        self.Y_input = QLineEdit()
        self.Y_input.setValidator(QDoubleValidator(0.01, 1000.0, 2))  # То же для оси Y
        layout.addWidget(self.Y_label)
        layout.addWidget(self.Y_input)

        # Кнопка
        self.next_button = QPushButton("Далее")
        self.next_button.clicked.connect(self.validate_input)
        layout.addWidget(self.next_button)

        self.setLayout(layout)

    def validate_input(self):
        gender = self.gender_input.currentText()
        age_text = self.age_input.text().strip()
        X_text = self.X_input.text().strip()
        Y_text = self.Y_input.text().strip()

        if gender not in ["М", "Ж"]:
            QMessageBox.warning(self, "Ошибка", "Пол должен быть 'М' или 'Ж'.")
            return

        if not age_text.isdigit() or int(age_text) < 0:
            QMessageBox.warning(self, "Ошибка", "Возраст должен быть целым числом от 0 и выше.")
            return

        try:
            X_text = X_text.replace(',', '.')
            X_value = float(X_text)
            if X_value <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Ось X должна быть положительным числом.")
            return

        try:
            Y_text = Y_text.replace(',', '.')
            Y_value = float(Y_text)
            if Y_value <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Ось Y должна быть положительным числом.")
            return

        self.patient_data = {
            "gender": gender,
            "age": int(age_text),
            "FOV": (X_value, Y_value)
        }

        # Переход к следующему окну
        self.image_upload_window = ImageUploadWindow(
            self.patient_data,
            diameter_model=self.diameter_model,
            position_model=self.position_model
        )
        self.image_upload_window.show()
        self.close()
