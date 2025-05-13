import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from ui.diameter_edit_window import DiameterEditWindow
from ui.results_window import ResultsWindow
from ui.positions_view_window import PositionsViewWindow

@pytest.fixture(scope="session")
def app():
    return QApplication([])

def test_diameter_edit_window_save(qtbot):
    modified_diameters = [(10.0, 12.0)]
    updated = []

    def update_callback():
        updated.append(True)

    window = DiameterEditWindow(modified_diameters, update_callback, image_paths=["tests/img-00040-00004.jpg"], fov_x=20, fov_y=20, img_width=512, img_height=512)
    qtbot.addWidget(window)

    l_input, r_input = window.inputs[0]
    l_input.setText("2.5")
    r_input.setText("3.0")

    qtbot.mouseClick(window.save_button, Qt.LeftButton)

    assert updated
    new_left, new_right = modified_diameters[0]
    assert 0 < new_left < 100
    assert 0 < new_right < 100

def test_results_window_text_generation(qtbot):
    patient_data = {"age": 45, "gender": "М", "FOV": (20, 20)}
    image_paths = ["test.png"]
    predicted_diameters = [(10.0, 12.0)]
    predicted_positions = [[(100, 200), (300, 400)]]

    window = ResultsWindow(patient_data, image_paths, predicted_diameters, predicted_positions)
    qtbot.addWidget(window)
    text = window.result_text.toPlainText()

    assert "Диаметры (мм)" in text
    assert "Левый сосуд" in text
    assert "Правый сосуд" in text

def test_positions_view_window_opens(qtbot):
    def update_callback(new_positions): pass

    window = PositionsViewWindow(
        image_paths=["tests/img-00040-00004.jpg"],
        predicted_positions=[[(100, 100), (200, 200)]],
        diameters_px=[(5.0, 5.0)],
        update_callback=update_callback
    )
    qtbot.addWidget(window)
    window.show()

    assert window.isVisible()

