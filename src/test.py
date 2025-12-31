import numpy as np
import tensorflow as tf
import os
import glob
import pandas as pd
import sys
import pickle 
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "experiments", "transformer_model")
SCALER_PATH = os.path.join(ROOT, "experiments", "scaler.pkl")
TEST_DATA_PATH = os.path.join(ROOT, "data", "test_keypoints")
TEST_OUTPUT_PATH = os.path.join(ROOT, "data", "output")

WINDOW_SIZE = 30 
TARGET_FPS = 30 

# [핵심 설정] 스무딩 강도 조절 (이 숫자를 바꾸세요!)
# 5 ~ 9 : 약한 보정 (원본 느낌 강함, 약간 떨림)
# 11 ~ 21 : 적절한 부드러움 (추천)
# 31 이상 : 매우 부드러움 (물속에서 움직이는 느낌이 날 수도 있음)
# ★중요: 무조건 '홀수'여야 합니다.
SMOOTHING_WINDOW = 15  
SMOOTHING_POLY = 2     # 곡선 차수 (2 또는 3 추천)

def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)

def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

def apply_smoothing(data_3d, window_length=15, polyorder=2):
    """
    Savitzky-Golay 필터를 사용하여 떨림을 제거하고 곡선을 부드럽게 만듭니다.
    """
    T, V, C = data_3d.shape 
    
    # 데이터가 너무 짧으면 스무딩 불가
    if T <= window_length: 
        return data_3d

    smoothed = np.zeros_like(data_3d)
    
    # 각 관절(V)의 각 좌표(x,y,z)마다 필터 적용
    for v in range(V):
        for c in range(C):
            smoothed[:, v, c] = savgol_filter(data_3d[:, v, c], window_length, polyorder)
            
    return smoothed

def upsample_sequence(data_3d, original_fps=30, target_fps=30):
    T, V, C = data_3d.shape
    duration = T / original_fps
    
    original_times = np.linspace(0, duration, T)
    new_T = int(duration * target_fps)
    new_times = np.linspace(0, duration, new_T)
    
    upsampled_data = np.zeros((new_T, V, C))
    
    for v in range(V):
        for c in range(C):
            # linear로 하면 딱딱 끊기지만 발이 안 미끄러짐
            # cubic으로 하면 부드럽지만 오버슈팅 발생
            # -> 해결책: Linear로 뽑고 나중에 Smoothing을 한 번 더 먹임
            f = interp1d(original_times, data_3d[:, v, c], kind='linear', fill_value="extrapolate")
            upsampled_data[:, v, c] = f(new_times)
            
    return upsampled_data

def align_to_floor(data_3d):
    feet_indices = [29, 30, 31, 32] 
    feet_y_values = data_3d[:, feet_indices, 1]
    
    floor_level = np.percentile(feet_y_values, 1)
    print(f"🔧 Floor Calibration: Shifting Up by {-floor_level:.4f}m")
    
    data_3d[:, :, 1] -= floor_level
    return data_3d

def refine_sequence(model, scaler, raw_path):
    raw = np.load(raw_path) 
    T = raw.shape[0]
    if T < WINDOW_SIZE: return None

    # Root 추출
    left_hip = raw[:, 23, :]
    right_hip = raw[:, 24, :]
    root_position = (left_hip + right_hip) / 2.0 
    root_position = root_position[:, np.newaxis, :] 

    centered_raw = raw - root_position
    raw_flat = centered_raw.reshape(T, -1) 

    mean = scaler["mean"]
    std = scaler["std"]
    part1 = raw_flat[:WINDOW_SIZE-1] 

    input_batch = []
    for i in range(T - WINDOW_SIZE + 1):
        seq = raw_flat[i : i + WINDOW_SIZE] 
        input_batch.append(seq)
    
    input_batch = np.array(input_batch) 
    input_batch = (input_batch - mean) / std 

    preds_norm = model.predict(input_batch, batch_size=256, verbose=0) 
    preds_last_frame = preds_norm[:, -1, :]

    part2 = preds_last_frame * std + mean
    
    refined_flat = np.concatenate([part1, part2], axis=0)
    refined_centered = refined_flat.reshape(-1, 33, 3)

    min_len = min(refined_centered.shape[0], root_position.shape[0])
    
    # 1. 원본 복구
    refined_final = refined_centered[:min_len] + root_position[:min_len]
    
    # 2. 바닥 보정
    refined_final = align_to_floor(refined_final)
    
    # 3. [1차] 기본 스무딩 (노이즈 제거용)
    refined_final = apply_smoothing(refined_final, window_length=9, polyorder=2)

    # 4. 프레임 정리 (Upsampling) - 여기서 Linear 보간 사용
    refined_upsampled = upsample_sequence(refined_final, original_fps=30, target_fps=TARGET_FPS)

    # 5. [★핵심] 최종 결과물에 강력한 스무딩 한 번 더 적용 (버터처럼 만들기)
    # Linear 보간으로 생긴 각진 부분을 둥글게 깎아줌
    print(f"✨ Applying Strong Smoothing (Window: {SMOOTHING_WINDOW})...")
    refined_upsampled = apply_smoothing(refined_upsampled, window_length=SMOOTHING_WINDOW, polyorder=SMOOTHING_POLY)

    return refined_upsampled

def main():
    os.makedirs(TEST_OUTPUT_PATH, exist_ok=True)
    files = glob.glob(f"{TEST_DATA_PATH}/*_raw.npy")

    if len(files) == 0:
        print("❌ No files found.")
        return

    try:
        model = load_trained_model()
        scaler = load_scaler()
    except Exception as e:
        print(f"Error: {e}")
        return

    for raw_path in files:
        base_name = os.path.basename(raw_path)
        name_only = base_name.replace("_raw.npy", "")
        print(f"Processing {base_name}...", end=" ")
        
        try:
            refined_data = refine_sequence(model, scaler, raw_path) 
            if refined_data is None: continue

            rows = []
            T, nBones, _ = refined_data.shape
            for t in range(T):
                for b in range(nBones):
                    x, y, z = refined_data[t, b]
                    rows.append([t, b, x, y, z, 1.0])

            df = pd.DataFrame(rows, columns=["frame", "landmark", "x", "y", "z", "visibility"])
            
            # 파일명에 _smooth 추가
            csv_path = f"{TEST_OUTPUT_PATH}/final_{name_only}_smooth.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved!")
            
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    main()