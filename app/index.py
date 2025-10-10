import sys

 
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget
from gui.home import HomeScreen
from gui.predict import PredictScreen
from gui.learn import LearnScreen

APP_NAME = "Cle Test"

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setFixedSize(400, 650)

        self.stack = QStackedWidget()

        # Instanciando telas
        self.home = HomeScreen(self.change_screen)
        self.predict = PredictScreen(self.change_screen)
        self.learn = LearnScreen(self.change_screen)

        # Adicionando telas ao stack
        self.stack.addWidget(self.home)  # index 0
        self.stack.addWidget(self.predict)  # index 1
        self.stack.addWidget(self.learn)  # index 2

        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stack)
        main_layout.setContentsMargins(0, 0, 0, 0)  # esquerda, topo, direita, baixo
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

    def change_screen(self, index):
        self.stack.setCurrentIndex(index)  # troca a tela mostrada

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())