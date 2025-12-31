# 02_process_height_dataset.py (최종 수정: 파일명 통일 버전)
import os
import sys
import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# 프로젝트 루트 경로 설정
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")

# 저장할 폴더가 없으면 생성
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

def load_data():
    # 1. 학습에 사용할 데이터(.npy)를 찾습니다.
    # 보통 'data/dataset'이나 'data/test_keypoints'에 있는 파일을 씁니다.
    # 여기서는 data 폴더 하위의 모든 _raw.npy를 찾도록 설정했습니다.
    search_path = os.path.join(DATA_DIR, "**", "*_raw.npy")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print(f"❌ Error: No .npy files found in {DATA_DIR}")
        print("Please put your training data (e.g., shuffle dance .npy) in 'data/dataset' folder.")
        # 만약 파일이 없으면 빈 배열 반환 방지
        return None

    all_data = []
    print(f"🔍 Found {len(files)} files for training.")

    for f in files:
        try:
            data = np.load(f) # (T, 33, 3)
            # 데이터가 너무 짧으면 무시 (30프레임 미만)
            if data.shape[0] < 30: continue
            
            # --- Root Centering (골반 0으로 맞추기) ---
            # 학습 데이터는 반드시 골반이 0이어야 함
            left_hip = data[:, 23, :]
            right_hip = data[:, 24, :]
            root = (left_hip + right_hip) / 2.0
            data_centered = data - root[:, np.newaxis, :]
            
            all_data.append(data_centered)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    return all_data

def create_windows(data_list, window_size=30):
    X_list = []
    Y_list = []
    
    for vid_data in data_list:
        T, V, C = vid_data.shape
        # Flatten: (T, 99)
        flat_data = vid_data.reshape(T, -1)
        
        # Sliding Window
        for i in range(T - window_size):
            # 입력: 과거 30프레임
            X_list.append(flat_data[i : i+window_size])
            # 정답: 바로 다음 프레임 (또는 같은 프레임 복원)
            # 여기서는 Autoencoder 방식(입력=정답)을 쓰거나, Next Frame Prediction을 씀
            # 노이즈 제거가 목적이므로, 입력과 똑같은 깨끗한 데이터를 정답으로 둠
            Y_list.append(flat_data[i : i+window_size]) 

    return np.array(X_list), np.array(Y_list)

def main():
    print("🚀 Processing Dataset...")
    
    # 1. 데이터 로드
    data_list = load_data()
    if data_list is None or len(data_list) == 0:
        return

    # 2. 윈도우 생성
    print("✂️ Creating Windows (Sequence Cutting)...")
    X, Y = create_windows(data_list)
    print(f"Dataset Shape: {X.shape}")

    # 3. 정규화 (Normalization) & Scaler 저장
    print("📏 Normalizing...")
    # (N*30, 99)로 펼쳐서 스케일링 계산
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)
    
    mean = np.mean(X_flat, axis=0)
    std = np.std(X_flat, axis=0) + 1e-6 # 0 나누기 방지
    
    # 정규화 적용
    X_norm = (X - mean) / std
    Y_norm = (Y - mean) / std # 정답도 정규화

    # Scaler 저장 (test.py에서 씀)
    scaler_data = {"mean": mean, "std": std}
    with open(os.path.join(EXPERIMENTS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler_data, f)
    print(f"💾 Scaler saved to {EXPERIMENTS_DIR}/scaler.pkl")

    # 4. 데이터 저장 (파일명: combined_raw.npy / combined_target.npy)
    # train.py가 이 이름을 찾습니다!
    raw_path = os.path.join(PROCESSED_DIR, "combined_raw.npy")
    target_path = os.path.join(PROCESSED_DIR, "combined_target.npy")
    
    np.save(raw_path, X_norm)
    np.save(target_path, Y_norm)
    
    print(f"✅ Data saved to {raw_path}")
    print("Ready for training! Run 'python train.py' now.")

if __name__ == "__main__":
    main()