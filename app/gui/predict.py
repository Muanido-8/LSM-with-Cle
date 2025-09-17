from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)
from PySide6.QtCore import Qt, QTimer, QSize, QThread
from PySide6.QtGui import QPixmap, QIcon, QFont, QImage
import qtawesome as qta
import numpy as np
from gui.predict_worker import PredictWorker

class PredictScreen(QWidget):
    def __init__(self, switch_func):
        super().__init__()
        self.switch_func = switch_func
        self.setup_ui()
        self._setup_predict_thread()

    def _setup_predict_thread(self):
        self.thread = QThread()
        self.worker = PredictWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.start)
        self.worker.frame_ready.connect(self.update_frame)

        # self.thread.start()
    
    def setup_ui(self):        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.setAlignment(Qt.AlignTop)
        
        # --- Câmera como fundo ---
        self.camera_view = QLabel()
        self.camera_view.setFixedSize(400, 650)
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setText("Carregando câmera")
        self.camera_view.setStyleSheet("color: black; font-size: 32px; background-color: transparent;")
        
        # --- Overlay topo (close) ---
        top_overlay = QWidget(self.camera_view)
        top_overlay.setGeometry(0, 0, 400, 80)
        top_layout = QHBoxLayout(top_overlay)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        
        self.close_btn = QPushButton()
        self.close_btn.setIcon(qta.icon('mdi.close-circle', color='red'))
        self.close_btn.setIconSize(QSize(40, 40))
        self.close_btn.setStyleSheet("background-color: transparent; border: none;")
        self.close_btn.clicked.connect(self.on_close_click)
        top_layout.addWidget(self.close_btn, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        
        # --- Overlay inferior (3 ícones) ---
        bottom_overlay = QWidget(self.camera_view)
        bottom_overlay.setGeometry(0, 560, 400, 80)
        bottom_layout = QHBoxLayout(bottom_overlay)
        bottom_layout.setContentsMargins(20, 10, 20, 10)
        bottom_layout.setSpacing(20)
        bottom_layout.setAlignment(Qt.AlignCenter)
        
        self.top_label = QLabel("----")
        self.top_label.setWordWrap(True)
        self.top_label.setStyleSheet("background-color: white; color: black; border-radius: 6px; padding: 4px;")
        self.top_label.setFont(QFont("Avenir", 12))
        bottom_layout.addWidget(self.top_label, alignment=Qt.AlignVCenter)
        
        main_layout.addWidget(self.camera_view)
        self.setLayout(main_layout)
    
    # --- Função para atualizar frame do OpenCV ---
    def update_frame(self, qimg, label, conf):
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.camera_view.width(),
            self.camera_view.height(),
            Qt.KeepAspectRatio
        )
        self.camera_view.setPixmap(pixmap)
        self.top_label.setText(f"{label} ({conf:.2f})")
    
    def on_hidden(self):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()

    def on_visible(self):
        self.thread.start()
    
    def closeEvent(self, event):
        self.on_hidden()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.on_visible()

    def hideEvent(self, event):
        super().hideEvent(event)
        print("PredictScreen foi escondido")
        self.on_hidden()
    
    # --- Handlers ---
    def on_close_click(self):
        self.switch_func(0)
    
    def on_icon1_click(self):
        print("Icon 1 clicado")
    
    def on_icon2_click(self):
        print("Icon 2 clicado")
    
    def on_icon3_click(self):
        print("Icon 3 clicado")
