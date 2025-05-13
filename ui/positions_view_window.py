from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox,
    QSlider, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageDraw
from io import BytesIO


class PositionsViewWindow(QWidget):
    def __init__(self, image_paths, predicted_positions, diameters_px, update_callback=None):
        super().__init__()
        self.update_callback = update_callback
        self.setWindowTitle("Просмотр позиций артерий")
        self.setGeometry(300, 100, 600, 700)

        self.image_paths = image_paths
        self.predicted_positions = predicted_positions
        self.diameters_px = diameters_px
        self.current_index = 0
        self.modified_positions = [list(p) for p in predicted_positions]
        self.opacity = 255
        self.dragging_index = None

        self.init_ui()
        self.load_image()

    def init_ui(self):
        self.image_label = QLabel()
        self.image_label.setFixedSize(512, 512)
        self.image_label.setMouseTracking(True)

        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release

        image_layout = QHBoxLayout()
        image_layout.addStretch()
        image_layout.addWidget(self.image_label)
        image_layout.addStretch()

        self.prev_btn = QPushButton("Предыдущее изображение")
        self.prev_btn.clicked.connect(self.show_prev)

        self.next_btn = QPushButton("Следующее изображение")
        self.next_btn.clicked.connect(self.show_next)

        self.artery_selector = QComboBox()
        self.artery_selector.addItems(["Никакая", "Левая артерия", "Правая артерия"])
        self.artery_selector.currentIndexChanged.connect(self.set_dragging_index)

        self.save_btn = QPushButton("Сохранить позиции")
        self.save_btn.clicked.connect(self.save_positions)

        self.reset_btn = QPushButton("Вернуться к предсказанным позициям")
        self.reset_btn.clicked.connect(self.reset_positions)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(25)
        self.slider.setMaximum(255)
        self.slider.setValue(255)
        self.slider.valueChanged.connect(self.change_opacity)

        layout = QVBoxLayout()
        layout.addLayout(image_layout)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)

        edit_layout = QHBoxLayout()
        edit_layout.addWidget(QLabel("Выбор артерии:"))
        edit_layout.addWidget(self.artery_selector)

        layout.addLayout(nav_layout)
        layout.addLayout(edit_layout)
        layout.addWidget(QLabel("Прозрачность круга:"))
        layout.addWidget(self.slider)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.reset_btn)

        self.setLayout(layout)

    def load_image(self):
        img_path = self.image_paths[self.current_index]
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        draw = ImageDraw.Draw(img, "RGBA")

        positions = self.modified_positions[self.current_index]
        diameters = self.diameters_px[self.current_index]

        for (x, y), d in zip(positions, diameters):
            radius = d / 2
            cx, cy = x * 512, y * 512
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.ellipse(bbox, outline=(255, 255, 255, self.opacity), width=3)

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        qimg = QImage.fromData(buffer.getvalue(), "PNG")
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def show_next(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_image()

    def change_opacity(self, value):
        self.opacity = value
        self.load_image()

    def save_positions(self):
        if self.update_callback:
            self.update_callback(self.modified_positions)
        QMessageBox.information(self, "Сохранение", "Позиции успешно сохранены!")

    def reset_positions(self):
        self.modified_positions[self.current_index] = list(self.predicted_positions[self.current_index])
        self.load_image()

    def set_dragging_index(self, index):
        if index == 0:
            self.dragging_index = None  # "Никакая"
        else:
            self.dragging_index = index - 1  # "Левая артерия" -> 0, "Правая артерия" -> 1

    def mouse_move(self, event):
        if self.dragging_index is not None:
            x = max(0, min(event.x() / 512, 1))
            y = max(0, min(event.y() / 512, 1))
            self.modified_positions[self.current_index][self.dragging_index] = (x, y)
            self.load_image()

    def mouse_release(self, event):
        self.dragging_index = None
