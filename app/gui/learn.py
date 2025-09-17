from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton
)
from PySide6.QtCore import Qt, QSize, QObject, Slot
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QFont
import qtawesome as qta
from PySide6.QtWebChannel import QWebChannel

class Bridge(QObject):
    def __init__(self, screen_object):
        super().__init__()
        self.screen_object = screen_object

    @Slot()
    def backToHome(self):
        print("Voltando para home screen")
        if self.screen_object and self.screen_object.switch_func:
            self.screen_object.switch_func(0)

class LearnScreen(QWidget):
    def __init__(self, switch_func):
        super().__init__()
        self.switch_func = switch_func
        self.setup_ui()
    
    def setup_ui(self):
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.setAlignment(Qt.AlignTop)

        self.webview = QWebEngineView()
        self.webview.setFixedSize(400, 650)
        self.webview.setAttribute(Qt.WA_TranslucentBackground, True)
        self.webview.setStyleSheet("background: transparent;")
        self.webview.page().setBackgroundColor(Qt.transparent)

        # Configura QWebChannel
        self.channel = QWebChannel()
        self.bridge = Bridge(self)
        self.channel.registerObject("bridge", self.bridge)
        self.webview.page().setWebChannel(self.channel)

        self.webview.setUrl("http://127.0.0.1:5500/app/assets/render-object/index.html")  # coloque o path do HTML
        main_layout.addWidget(self.webview)
        
        self.setLayout(main_layout)

    def on_close_click(self):
        # Exemplo de callback
        if self.switch_func:
            self.switch_func(0)
