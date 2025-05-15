import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
import torch
from PyQt5.QtGui import QIcon

# Подключим модельные классы
from models.diameter_model import VertebralArteryRegressor
from models.position_model import ArteriesLocalizationModel
from ui.results_window import ResultsWindow

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diameter_model = VertebralArteryRegressor(pretrained=False).to(device)
    position_model = ArteriesLocalizationModel(pretrained=False).to(device)
    diameter_model.load_state_dict(torch.load('ui/weights/diameter_weights.pth', map_location=device))
    position_model.load_state_dict(torch.load('ui/weights/position_weights.pth', map_location=device))


    app = QApplication(sys.argv)
    main_window = MainWindow(diameter_model=diameter_model, position_model=position_model)
    app.setWindowIcon(QIcon('ui/icon/hse.ico'))
    main_window.setWindowIcon(QIcon('ui/icon/hse.ico'))
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()