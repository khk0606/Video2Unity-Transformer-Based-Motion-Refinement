import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# [설정] 파일 경로를 여기에 입력하세요
# ==================================================
# 1. Baseline 데이터 (Raw NPY 파일)
RAW_FILE_PATH = "/Users/ganghyeongyu/AI_ML_Python_Final/final_project/data/test_keypoints/AAAraw.npy" 

# 2. Student 데이터 (최종 생성된 CSV 파일)
REFINED_FILE_PATH = "/Users/ganghyeongyu/AI_ML_Python_Final/final_project/data/output/finalAAAsmoth.csv" 

# 3. 비교하고 싶은 구간 (프레임)
START_FRAME = 100
END_FRAME = 200
# ==================================================

def draw_comparison_graphs():
    # 1. 데이터 로드
    try:
        raw_data = np.load(RAW_FILE_PATH)  # (T, 33, 3)
        refined_df = pd.read_csv(REFINED_FILE_PATH)
    except Exception as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        return

    # CSV 데이터를 (T, 33, 3) 형태로 변환
    num_frames = refined_df['frame'].max() + 1
    refined_data = np.zeros((num_frames, 33, 3))
    
    for _, row in refined_df.iterrows():
        t, v = int(row['frame']), int(row['landmark'])
        refined_data[t, v, 0] = row['x']
        refined_data[t, v, 1] = row['y']
        refined_data[t, v, 2] = row['z']

    # 데이터 길이 맞추기
    min_len = min(len(raw_data), len(refined_data))
    raw_data = raw_data[:min_len]
    refined_data = refined_data[:min_len]

    # 비교 구간 설정
    t = np.arange(START_FRAME, END_FRAME)
    
    # -------------------------------------------------------
    # Graph 1: Hand Trajectory (Jitter Removal)
    # -------------------------------------------------------
    joint_idx = 15  # Left Wrist
    raw_y = raw_data[START_FRAME:END_FRAME, joint_idx, 1]
    ref_y = refined_data[START_FRAME:END_FRAME, joint_idx, 1]

    plt.figure(figsize=(12, 4))
    plt.title("Comparison of Joint Stability (Left Wrist Y-axis)", fontsize=14)
    plt.plot(t, raw_y, label='Baseline (Raw)', color='red', alpha=0.5, linestyle='--')
    plt.plot(t, ref_y, label='Student (Ours)', color='blue', linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Position Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() # 캡처해서 PPT에 넣기

    # -------------------------------------------------------
    # Graph 2: Foot Contact (Floor Alignment)
    # -------------------------------------------------------
    joint_idx = 29  # Left Heel
    raw_y = raw_data[START_FRAME:END_FRAME, joint_idx, 1]
    ref_y = refined_data[START_FRAME:END_FRAME, joint_idx, 1]

    plt.figure(figsize=(12, 4))
    plt.title("Foot Contact Stability (Left Heel Y-axis)", fontsize=14)
    plt.plot(t, raw_y, label='Baseline (Raw)', color='red', alpha=0.5, linestyle='--')
    plt.plot(t, ref_y, label='Student (Ours)', color='green', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5) # 바닥선
    plt.xlabel("Frame")
    plt.ylabel("Height (Y)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() # 캡처해서 PPT에 넣기

    # -------------------------------------------------------
    # Graph 3: Velocity Smoothness (Denoising Effect)
    # -------------------------------------------------------
    # 속도 = 현재 위치 - 이전 위치
    raw_vel = np.diff(raw_data[START_FRAME:END_FRAME, 0, 1]) # Nose Y 속도
    ref_vel = np.diff(refined_data[START_FRAME:END_FRAME, 0, 1])

    plt.figure(figsize=(12, 4))
    plt.title("Motion Smoothness (Velocity Change of Nose)", fontsize=14)
    plt.plot(t[:-1], raw_vel, label='Baseline (Raw)', color='red', alpha=0.5)
    plt.plot(t[:-1], ref_vel, label='Student (Ours)', color='purple', linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Velocity (Delta Y)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() # 캡처해서 PPT에 넣기

if __name__ == "__main__":
    draw_comparison_graphs()