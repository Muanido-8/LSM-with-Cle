import os
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# ========================
# 🔹 Configurações
# ========================

DATA_PATH = os.getenv("DATA_PATH", "C:/xampp/htdocs/LSM-with-Cle/train/data/proccessed")

# Definir ações
actions = np.array([
    "Nao",
    "Olá",
    "Bom"
])

sequence_length = 30  # Frames por vídeo
num_features = 21*3*2  # apenas mãos: 21 landmarks por mão, 3 coordenadas, 2 mãos = 126

# ========================
# 🔹 Carregar dataset (apenas mãos)
# ========================

sequences, labels = [], []
label_map = {label: num for num, label in enumerate(actions)}

for action in actions:
    print(f"--------------------> Processando {action}")
    action_path = os.path.join(DATA_PATH, action)
    for sequence in os.listdir(action_path):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(action_path, sequence, f"{frame_num}.npy")
            res = np.load(npy_path)
            # garante shape correto (126)
            if res.shape[0] != num_features:
                raise ValueError(f"Frame {npy_path} não tem o shape esperado de {num_features}")
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

# ========================
# 🔹 Criar modelo LSTM (apenas mãos)
# ========================

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, num_features)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Camadas densas
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))

# Camada de saída
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ========================
# 🔹 Callbacks
# ========================

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)
es_callback = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

# ========================
# 🔹 Treino
# ========================

model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[tb_callback, es_callback])

# ========================
# 🔹 Avaliação
# ========================

loss, acc = model.evaluate(X_test, y_test)
print(f"Acuracia: {acc:.2f}")

# ========================
# 🔹 Salvar modelo
# ========================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"model_lstm_hands_{timestamp}_{acc}.keras"
model_path = os.path.join("C:/xampp/htdocs/LSM-with-Cle/app/models", model_name)
model.save(model_path)
print(f"💾 Modelo salvo em {model_path}")
