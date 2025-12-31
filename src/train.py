import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# --- 경로 자동 설정 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(CURRENT_DIR, "data")):
    ROOT = CURRENT_DIR
elif os.path.exists(os.path.join(os.path.dirname(CURRENT_DIR), "data")):
    ROOT = os.path.dirname(CURRENT_DIR)
else:
    ROOT = os.path.dirname(CURRENT_DIR)
    print(f"⚠️ Warning: 'data' folder not found relative to {CURRENT_DIR}")

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODEL_SAVE_DIR = os.path.join(ROOT, "experiments", "transformer_model")
SCALER_PATH = os.path.join(ROOT, "experiments", "scaler.pkl")

os.makedirs(os.path.dirname(MODEL_SAVE_DIR), exist_ok=True)

# 하이퍼파라미터
WINDOW_SIZE = 30
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# --- 데이터 제너레이터 (노이즈 주입 포함) ---
class PoseDataGenerator(keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=32, shuffle=True, augment=False):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment 
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.X[idxs]
        Y_batch = self.Y[idxs]

        if self.augment:
            # 노이즈 주입
            noise = np.random.normal(0, 0.02, X_batch.shape)
            X_batch = X_batch + noise 

        return X_batch, Y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def build_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # 1. 차원 확장 (Embedding/Projection)
    # 입력을 99 -> 128로 늘립니다.
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x) 
    
    # --- Transformer Block ---
    # 2. Attention
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
    # (Skip Connection 1) 128 + 128 -> OK
    x = layers.Add()([x, attention_output]) 
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # 3. Feed Forward Network (FFN)
    ffn = layers.Dense(128, activation="relu")(x)
    
    # [수정된 부분] ★★★ 중요 ★★★
    # 기존 코드: ffn = layers.Dense(input_shape[-1])(ffn) -> 99로 줄여서 에러남
    # 수정 코드: ffn = layers.Dense(128)(ffn) -> 128 유지
    ffn = layers.Dense(128)(ffn) 
    
    # (Skip Connection 2) 128 + 128 -> OK! (이제 에러 안 남)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # 4. 최종 출력 (Output Projection)
    # 블록을 다 통과한 뒤에 마지막에 99(원본 크기)로 줄입니다.
    outputs = layers.Dense(input_shape[-1], activation="linear")(x)
    
    return keras.Model(inputs, outputs)

def main():
    print(f"🔍 Checking path: {PROCESSED_DIR}")
    
    raw_path = os.path.join(PROCESSED_DIR, "combined_raw.npy")
    target_path = os.path.join(PROCESSED_DIR, "combined_target.npy")
    
    if not os.path.exists(raw_path):
        print(f"❌ Error: File not found at {raw_path}")
        return

    print("✅ Data found! Loading...")
    X = np.load(raw_path) 
    Y = np.load(target_path) 
    
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    train_gen = PoseDataGenerator(X_train, Y_train, BATCH_SIZE, augment=True)
    val_gen = PoseDataGenerator(X_val, Y_val, BATCH_SIZE, augment=False)

    print("Building Model...")
    input_shape = (X.shape[1], X.shape[2]) 
    model = build_transformer_model(input_shape)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse", 
        metrics=["mae"]
    )
    
    print("Start Training...")
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_DIR, save_best_only=True, monitor="val_loss"
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )

    print("Training Done. Model saved.")

if __name__ == "__main__":
    main()