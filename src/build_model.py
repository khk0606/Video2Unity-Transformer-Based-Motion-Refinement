import tensorflow as tf
from tensorflow.keras import layers, models

def build_transformer_model(window_size=30, feature_dim=99, num_heads=4, ff_dim=128):
    """
    Transformer Encoder for Motion Correction
    
    [개선 사항]
    - Skip Connection(잔차 연결) 로직 수정: (x + x) -> (Original + Processed)
    - 이를 통해 원본 동작의 특징을 잃지 않고 노이즈만 제거하도록 학습 효율 증대
    
    Input: (Batch, Window, 99)
    Output: (Batch, 99) -> Current Frame Correction
    """
    
    # --- 1. 입력 단계 (Input) ---
    inputs = layers.Input(shape=(window_size, feature_dim))

    # --- 2. 임베딩 & 위치 정보 주입 (Embedding) ---
    # 입력을 feature_dim 크기로 투영 (데이터 특성 추출)
    x = layers.Dense(feature_dim)(inputs)
    
    # 시간 순서 정보 생성 (0 ~ window_size-1)
    positions = tf.range(start=0, limit=window_size, delta=1)
    pos_embedding = layers.Embedding(input_dim=window_size, output_dim=feature_dim)(positions)
    
    # 입력 + 위치 정보 더하기
    x = x + pos_embedding

    # --- 3. Transformer Encoder Block ---
    
    # [Skip Connection 1] - Attention 수행 전 상태 저장
    skip_1 = x 
    
    # Self-Attention: 프레임 간 관계 파악 (어느 시점이 중요한가?)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature_dim)(x, x)
    attention_output = layers.Dropout(0.1)(attention_output)
    
    # Add & Norm: 원본(skip_1) + 처리된 값(attention_output)
    x = layers.LayerNormalization(epsilon=1e-6)(skip_1 + attention_output)

    # [Skip Connection 2] - FFN 수행 전 상태 저장
    skip_2 = x
    
    # Feed Forward Network (FFN)
    ffn_output = layers.Dense(ff_dim, activation="gelu")(x) # 활성화 함수 GELU 사용
    ffn_output = layers.Dense(feature_dim)(ffn_output)      # 차원 복구
    ffn_output = layers.Dropout(0.1)(ffn_output)
    
    # Add & Norm: 원본(skip_2) + 처리된 값(ffn_output)
    # (기존 코드의 x + x 버그 수정됨)
    x = layers.LayerNormalization(epsilon=1e-6)(skip_2 + ffn_output) 

    # --- 4. 출력 단계 (Output Head) ---
    # 시퀀스 데이터(30프레임)를 하나의 벡터(현재 프레임)로 압축
    x = layers.Flatten()(x) 
    
    # 최종 출력: (Batch, 99) - 보정된 33개 관절의 x,y,z 좌표
    outputs = layers.Dense(feature_dim)(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Motion_Transformer")
    return model