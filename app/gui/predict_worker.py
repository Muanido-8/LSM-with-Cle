import cv2
import numpy as np
import random
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage
import mediapipe as mp

from keras.models import load_model
model = load_model("C:/xampp/htdocs/Traducao gestos/app/models/model_lstm_hands_20250917_231208_1.0.keras")   # <-- substitui pelo caminho do teu modelo
actions = np.array([
    "Domingo",
    "Explicar",
    "Nao",
    "Ola",
    "Professor",
    "Correto"
])  # <-- lista das tuas classes


class PredictWorker(QObject):
    frame_ready = Signal(QImage, str, float)  
    # envia imagem + label + confidence

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.running = False

        # Variáveis da predição
        self.sequence = []
        self.predictions = []
        self.sentence = []
        self.threshold = 0.5

        # Mediapipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    # ---------- Funções auxiliares ---------- 40736

    def mediapipe_detection(self, image, model):
        """Aplica MediaPipe e retorna imagem + resultados"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        """Desenha landmarks"""
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2),
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2),
            )

    def extract_keypoints(self, results):
        lh = np.array([[res.x, res.y, res.z] 
                       for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] 
                       for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])

    # ---------- Loop principal ----------

    def start(self):
        self.running = True
        self._run()

    def stop(self):
        self.running = False

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)

        with self.mp_holistic.Holistic(min_detection_confidence=0.5,  min_tracking_confidence=0.5) as holistic:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Espelha (modo selfie)
                # frame = cv2.flip(frame, 1)

                # Detection
                image, results = self.mediapipe_detection(frame, holistic)
                self.draw_styled_landmarks(image, results)

                # Prediction
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]

                label, conf = "", 0.0
                if len(self.sequence) == 30:
                    res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    self.predictions.append(np.argmax(res))

                    if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > self.threshold:
                            if len(self.sentence) == 0 or actions[np.argmax(res)] != self.sentence[-1]:
                                self.sentence.append(actions[np.argmax(res)])

                    if len(self.sentence) > 5:
                        self.sentence = self.sentence[-5:]

                    label = actions[np.argmax(res)]
                    conf = float(res[np.argmax(res)])

                # Converte para QImage
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

                self.frame_ready.emit(qimg, label, conf)

        cap.release()
