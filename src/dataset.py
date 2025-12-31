# dataset.py
import os
import numpy as np
from glob import glob
import sys
import pickle # Scaler 저장용

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
SCALER_PATH = os.path.join(ROOT, "experiments", "scaler.pkl") # 저장 경로

# Window Size (과거 30프레임 참조)
WINDOW_SIZE = 30 

def create_sequences(data, target, window_size):
    """
    Sliding Window 방식으로 시퀀스 데이터 생성
    """
    X_seq, Y_seq = [], []
    # 데이터가 윈도우보다 작으면 스킵
    if len(data) <= window_size:
        return [], []
        
    for i in range(len(data) - window_size):
        # 입력: i ~ i+window_size (30프레임)
        X_seq.append(data[i : i + window_size])
        # 정답: window의 마지막 프레임 (현재 시점)
        Y_seq.append(target[i + window_size - 1])
        
    return X_seq, Y_seq

def load_dataset():
    raw_list = sorted(glob(f"{PROCESSED_DIR}/*_raw.npy"))
    X_all, Y_all = [], []

    print(f"Loading data from {PROCESSED_DIR}...")
    
    # 1. 모든 데이터 로드 및 시퀀싱
    for rp in raw_list:
        tp = rp.replace("_raw.npy", "_target.npy")
        if not os.path.exists(tp):
            print(f"Target not found for {rp}")
            continue

        raw = np.load(rp) # (T, 33, 3)
        tgt = np.load(tp) # (T, 33, 3)

        # Flatten (T, 99)
        raw = raw.reshape(len(raw), -1)
        tgt = tgt.reshape(len(tgt), -1)

        x_seq, y_seq = create_sequences(raw, tgt, WINDOW_SIZE)
        
        if len(x_seq) > 0:
            X_all.extend(x_seq)
            Y_all.extend(y_seq)

    if len(X_all) == 0:
        print("No valid data found!")
        return None, None

    X = np.array(X_all) # (N, 30, 99)
    Y = np.array(Y_all) # (N, 99)

    print(f"Raw Dataset Shape: X={X.shape}, Y={Y.shape}")

    # 2. 데이터 정규화 (Normalization)
    # 학습 데이터 전체의 평균과 표준편차 계산
    print("Computing Mean & Std for Normalization...")
    # X shape: (samples, time, features) -> features(2번축) 별로 통계 계산
    mean = X.mean(axis=(0, 1)) # (99,)
    std = X.std(axis=(0, 1)) + 1e-6 # (99,) division by zero 방지

    # 정규화 적용 (Z-Score)
    X = (X - mean) / std
    # Y(타겟)도 좌표값이므로 동일한 스케일로 변환해야 학습이 잘됨
    Y = (Y - mean) / std 

    # 3. Scaler 저장 (Test 때 쓰기 위해)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)
    print(f"Scaler saved -> {SCALER_PATH}")

    print("Data Normalization Done.")
    return X, Y
