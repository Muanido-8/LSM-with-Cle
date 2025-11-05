import os
import numpy as np
import cv2
import mediapipe as mp

# ========================
# 🔹 Configurações
# ========================

# Carregar variáveis do .env
DATA_PATH = os.getenv("DATA_PATH", "C:/xampp/htdocs/LSM-with-Cle/train/data/proccessed")
VIDEO_PATH = os.getenv("VIDEO_PATH", "C:/xampp/htdocs/LSM-with-Cle/train/data/videos")

# Definir ações (edite depois conforme seus gestos)
actions = np.array([
    "Nao",
    "Olá",
    "Bom"
])

# Quantos frames vamos extrair de cada vídeo
sequence_length = 30

# ========================
# 🔹 MediaPipe setup
# ========================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return image, results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# ========================
# 🔹 Pipeline principal
# ========================
if __name__ == "__main__":
    os.makedirs(DATA_PATH, exist_ok=True)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            action_video_path = os.path.join(VIDEO_PATH, action)
            action_data_path = os.path.join(DATA_PATH, action)
            os.makedirs(action_data_path, exist_ok=True)

            for idx, video_file in enumerate(os.listdir(action_video_path)):
                print(f"/n▶ Processando {video_file} da ação {action}")

                cap = cv2.VideoCapture(os.path.join(action_video_path, video_file))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Selecionar frames uniformemente
                frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

                for frame_num, frame_index in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"⚠ Não foi possível ler frame {frame_index}")
                        continue

                    _, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)

                    # Nome do arquivo .npy: video_indice_frame.npy
                    seq_folder = os.path.join(action_data_path, str(idx))
                    os.makedirs(seq_folder, exist_ok=True)
                    np.save(os.path.join(seq_folder, f"{frame_num}.npy"), keypoints)

                cap.release()
                print(f"✅ Concluído {video_file}")
