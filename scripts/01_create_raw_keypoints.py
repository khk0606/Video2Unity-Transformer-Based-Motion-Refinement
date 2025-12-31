# 01_create_raw_keypoints.py (Safari 버전)

import os
import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
import yt_dlp 

# 프로젝트 경로 설정
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
RAW_KEYPOINT_DIR = os.path.join(ROOT, "data", "raw_keypoints")

os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
os.makedirs(RAW_KEYPOINT_DIR, exist_ok=True)

# Mediapipe 설정
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def download_youtube(url):
    print(f"Trying to download: {url}")
    
    # 현재 실행 중인 파일(스크립트)의 위치를 기준으로 cookies.txt 찾기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cookie_path = os.path.join(script_dir, "cookies.txt")

    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': os.path.join(RAW_VIDEO_DIR, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        
        # [핵심] 키체인 인증 대신, 아까 만든 파일을 직접 읽게 합니다.
        'cookiefile': cookie_path, 
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # (나머지 코드는 그대로)
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            safe_title = info['title'].replace(" ", "_").replace("/", "_")
            print(f"🎬 Downloaded → {video_path}")
            return video_path, safe_title
            
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return None, None

def extract_3d_keypoints(video_path, dir_path=RAW_KEYPOINT_DIR, name="Data"):
    if video_path is None: return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_rows = []
    pbar = tqdm(total=total_frames, desc=f"Extracting {name}", ascii=True, dynamic_ncols=False)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_model.process(rgb)

        if res.pose_world_landmarks:
            for i, lm in enumerate(res.pose_world_landmarks.landmark):
                pose_rows.append({
                    "frame": frame_idx,
                    "landmark": i,
                    "x": lm.x, "y": -lm.y, "z": lm.z,
                    "visibility": lm.visibility,
                })

        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()

    if len(pose_rows) == 0: return

    df = pd.DataFrame(pose_rows)
    out_path = f"{dir_path}/{name}.npz"
    np.savez(out_path, data=df.to_numpy())
    print(f"📌 Saved 3D keypoints → {out_path}")

def main():
    url_list = [
        "https://www.youtube.com/shorts/uZXxJUTAbps",
        "https://www.youtube.com/shorts/N2Ie3RCpSqI",
        "https://www.youtube.com/shorts/nL4brQI1J6A",
        "https://www.youtube.com/shorts/edROOVlmDx0",
        "https://www.youtube.com/shorts/72nLd87HZ48",
    ]

    for url in url_list:
        try:
            video_path, name = download_youtube(url)
            if video_path:
                extract_3d_keypoints(video_path, RAW_KEYPOINT_DIR, name)
        except Exception as e:
            print(f"Skipping {url} due to error: {e}")

if __name__ == "__main__":
    main()