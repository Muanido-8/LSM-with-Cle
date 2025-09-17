from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
import os

class HomeScreen(QWidget):
    def __init__(self, switch_func):
        super().__init__()
        self.switch_func = switch_func
        self.setup_ui()
    
    def setup_ui(self):
        # Layout principal
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 40, 20, 40)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        
        # Título
        title = QLabel("Bem-vindo")
        title_font = QFont("Arial", 24, QFont.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Logo abaixo do título
        pixmap_path = os.path.join(os.path.dirname(__file__), "../assets/logo.png")
        pixmap = QPixmap(pixmap_path)
        logo = QLabel()
        pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)
        
        layout.addStretch(1)  # empurra os botões para baixo
        
        # Botões
        self.translate_btn = QPushButton("Traduzir")
        self.learn_btn = QPushButton("Aprender")
        
        # Estilo dos botões (Bootstrap primary e outline)
        primary_color = "#0d6efd"  # azul Bootstrap
        button_style = f"""
            QPushButton {{
                background-color: {primary_color};
                color: white;
                border-radius: 12px;
                padding: 14px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: #0b5ed7;
            }}
            QPushButton:pressed {{
                background-color: #0a58ca;
            }}
        """
        outline_style = f"""
            QPushButton {{
                background-color: transparent;
                color: {primary_color};
                border: 2px solid {primary_color};
                border-radius: 12px;
                padding: 14px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: {primary_color};
                color: white;
            }}
            QPushButton:pressed {{
                background-color: #0a58ca;
                color: white;
            }}
        """
        self.translate_btn.setStyleSheet(button_style)
        self.learn_btn.setStyleSheet(outline_style)
        
        # Tamanho dos botões
        self.translate_btn.setFixedHeight(50)
        self.learn_btn.setFixedHeight(50)
        self.translate_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.learn_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Conectando handlers
        self.translate_btn.clicked.connect(self.on_translate_click)
        self.learn_btn.clicked.connect(self.on_learn_click)
        
        # Adicionando botões ao layout
        layout.addWidget(self.translate_btn)
        layout.addWidget(self.learn_btn)
        
        self.setLayout(layout)
    
    # Handlers
    def on_translate_click(self):
        self.switch_func(1)
    
    def on_learn_click(self):
        self.switch_func(2)
