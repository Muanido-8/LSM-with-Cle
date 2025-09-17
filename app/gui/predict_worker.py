import cv2
import numpy as np
import random
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage
import mediapipe as mp

from keras.models import load_model
model = load_model("C:/xampp/htdocs/Traducao gestos/app/models/model_lstm_20250915_113408.keras")   # <-- substitui pelo caminho do teu modelo
actions = np.array(["ola", "nao", "z"])  # <-- lista das tuas classes


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
        self.colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in actions]

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
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2),
            )
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1),
            )
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
        """Extrai keypoints em vetor fixo"""
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                         for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] 
                         for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] 
                       for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] 
                       for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def prob_viz(self, res, actions, input_frame, colors):
        """Barra de probabilidade"""
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            color = colors[num]
            cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), color, -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return output_frame

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

                    # image = self.prob_viz(res, actions, image, self.colors)
                    label = actions[np.argmax(res)]
                    conf = float(res[np.argmax(res)])

                # Converte para QImage
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

                self.frame_ready.emit(qimg, label, conf)

        cap.release()
