import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==================================================
# [설정] 실험 파라미터
# ==================================================
WINDOW_SIZE = 30    # Transformer가 보는 길이
FEATURE_DIM = 99    # 관절 좌표 수 (33 * 3)
EPOCHS = 15         # 비교를 위해 짧게 학습
BATCH_SIZE = 32
DATA_SAMPLES = 1000 # 가상 데이터 개수

# ==================================================
# 1. Baseline Model: MLP (User Provided)
# ==================================================
def build_baseline_mlp(input_dim=99, output_dim=99):
    # 제공해주신 코드 그대로 사용
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),  # (Batch, 99) - 단일 프레임
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(output_dim)           # (Batch, 99)
    ], name="Baseline_MLP")
    return model

# ==================================================
# 2. Student Model: Transformer (Ours)
# ==================================================
def build_student_transformer(window_size=30, feature_dim=99):
    inputs = layers.Input(shape=(window_size, feature_dim)) # (Batch, 30, 99)
    
    # Embedding & Positional Encoding
    x = layers.Dense(feature_dim)(inputs)
    positions = tf.range(start=0, limit=window_size, delta=1)
    pos_embedding = layers.Embedding(input_dim=window_size, output_dim=feature_dim)(positions)
    x = x + pos_embedding
    
    # Transformer Encoder Block (Simplified)
    # Self-Attention: 시간적 문맥 학습
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=feature_dim)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # FFN
    ffn_output = layers.Dense(128, activation="gelu")(x)
    ffn_output = layers.Dense(feature_dim)(ffn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    # Output Head (Sequence -> Single Frame Prediction)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(feature_dim)(x)
    
    model = models.Model(inputs, outputs, name="Student_Transformer")
    return model

# ==================================================
# 3. 데이터 생성 및 학습 비교
# ==================================================
def main():
    # -------------------------------------------------------
    # (1) 가상 데이터 생성 (실제 데이터가 없어도 그래프 생성 가능)
    # -------------------------------------------------------
    # X_seq: (1000, 30, 99) -> Transformer용 (시퀀스)
    X_seq = np.random.randn(DATA_SAMPLES, WINDOW_SIZE, FEATURE_DIM).astype(np.float32)
    
    # X_single: (1000, 99) -> MLP용 (시퀀스의 가운데 프레임 하나만 떼어냄)
    middle_idx = WINDOW_SIZE // 2
    X_single = X_seq[:, middle_idx, :] 
    
    # Y_target: (1000, 99) -> 정답 (노이즈 없는 깨끗한 데이터 가정)
    Y_target = np.random.randn(DATA_SAMPLES, FEATURE_DIM).astype(np.float32)

    # -------------------------------------------------------
    # (2) Baseline MLP 학습
    # -------------------------------------------------------
    print("📉 Training Baseline MLP (Single Frame)...")
    mlp = build_baseline_mlp(input_dim=FEATURE_DIM, output_dim=FEATURE_DIM)
    mlp.compile(optimizer='adam', loss='mse')
    hist_mlp = mlp.fit(X_single, Y_target, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, validation_split=0.2)

    # -------------------------------------------------------
    # (3) Student Transformer 학습
    # -------------------------------------------------------
    print("📈 Training Student Transformer (Sequence)...")
    transformer = build_student_transformer(window_size=WINDOW_SIZE, feature_dim=FEATURE_DIM)
    transformer.compile(optimizer='adam', loss='mse')
    hist_trans = transformer.fit(X_seq, Y_target, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, validation_split=0.2)

    # -------------------------------------------------------
    # (4) 결과 비교 그래프 그리기
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # MLP Loss (빨간 점선)
    plt.plot(hist_mlp.history['val_loss'], 'r--', label='Baseline MLP (Single Frame)', linewidth=2)
    
    # Transformer Loss (파란 실선)
    plt.plot(hist_trans.history['val_loss'], 'b-', label='Student Transformer (30 Frames)', linewidth=3)
    
    plt.title("Architecture Comparison: MLP vs Transformer", fontsize=16, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Validation Loss (MSE)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✅ 비교 그래프 생성 완료!")

if __name__ == "__main__":
    main()