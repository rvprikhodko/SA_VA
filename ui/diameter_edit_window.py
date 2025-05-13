from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit,
    QPushButton, QMessageBox, QDialog, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PIL import Image
from io import BytesIO


class DiameterEditWindow(QWidget):
    def __init__(self, diameters, update_callback, fov_x, fov_y, img_width, img_height, image_paths):
        super().__init__()
        self.setWindowTitle("Редактирование диаметров артерий")
        self.setGeometry(300, 300, 500, 400)

        self.diameters_px = diameters  # в пикселях
        self.update_callback = update_callback
        self.image_paths = image_paths
        self.inputs = []

        # Расчет пересчета px → mm
        self.px_to_mm_x = (fov_x * 10) / img_width
        self.px_to_mm_y = (fov_y * 10) / img_height
        self.px_to_mm = (self.px_to_mm_x + self.px_to_mm_y) / 2
        self.mm_to_px = 1 / self.px_to_mm

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Введите новые значения диаметров (в мм):"))

        for i, (left_px, right_px) in enumerate(self.diameters_px):
            left_mm = round(left_px * self.px_to_mm, 2)
            right_mm = round(right_px * self.px_to_mm, 2)

            row = QHBoxLayout()
            l_edit = QLineEdit(str(left_mm))
            r_edit = QLineEdit(str(right_mm))
            self.inputs.append((l_edit, r_edit))

            img_btn = QPushButton("Просмотр изображения")
            img_btn.clicked.connect(lambda _, idx=i: self.show_image(idx))

            row.addWidget(QLabel(f"Изображение {i+1} — Левый:"))
            row.addWidget(l_edit)
            row.addWidget(QLabel("Правый:"))
            row.addWidget(r_edit)
            row.addWidget(img_btn)

            layout.addLayout(row)

        self.save_button = QPushButton("Сохранить изменения")
        self.save_button.clicked.connect(self.save_diameters)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def show_image(self, index):
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert("RGB").resize((512, 512))

        byte_io = BytesIO()
        img.save(byte_io, format="PNG")
        pixmap = QPixmap()
        pixmap.loadFromData(byte_io.getvalue())

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Изображение {index + 1}")
        dialog.setGeometry(350, 350, 600, 700)

        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignCenter)

        # Получаем соответствующие поля ввода из DiameterEditWindow
        l_edit_main, r_edit_main = self.inputs[index]

        # Создаём копии для редактирования
        l_edit = QLineEdit(l_edit_main.text())
        r_edit = QLineEdit(r_edit_main.text())

        # Привязка: при изменении копии — обновляется оригинал
        def sync_fields():
            l_val = l_edit.text().strip()
            r_val = r_edit.text().strip()
            l_edit_main.setText(l_val)
            r_edit_main.setText(r_val)

        l_edit.textChanged.connect(sync_fields)
        r_edit.textChanged.connect(sync_fields)

        edit_layout = QHBoxLayout()
        edit_layout.addWidget(QLabel("Левый (мм):"))
        edit_layout.addWidget(l_edit)
        edit_layout.addWidget(QLabel("Правый (мм):"))
        edit_layout.addWidget(r_edit)

        layout = QVBoxLayout()
        layout.addWidget(img_label)
        layout.addLayout(edit_layout)

        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def save_diameters(self):
        new_diameters_px = []
        try:
            for l_edit, r_edit in self.inputs:
                l_mm = float(l_edit.text())
                r_mm = float(r_edit.text())
                if l_mm <= 0 or r_mm <= 0:
                    raise ValueError("Диаметры должны быть положительными.")

                l_px = round(l_mm * self.mm_to_px, 2)
                r_px = round(r_mm * self.mm_to_px, 2)
                new_diameters_px.append((l_px, r_px))

        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return

        self.diameters_px[:] = new_diameters_px
        self.update_callback()
        self.close()
