import os
import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
import yt_dlp 

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
RAW_TEST_DIR = os.path.join(ROOT, "data", "test_keypoints")

os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
os.makedirs(RAW_TEST_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- [설정 수정] 민감도 극소화 (0.2) ---
# 0.5도 크다면 0.2로 줄입니다. 
# 이제 거의 제자리에서 무게중심만 살짝살짝 이동하는 느낌이 됩니다.
MOVE_SENSITIVITY_X = 0.2  
MOVE_SENSITIVITY_Z = 0.1  # 앞뒤 이동은 거의 0에 가깝게

def download_youtube(url):
    print(f"Trying to download: {url}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cookie_path = os.path.join(script_dir, "cookies.txt")

    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': os.path.join(RAW_VIDEO_DIR, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    if os.path.exists(cookie_path):
        ydl_opts['cookiefile'] = cookie_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            safe_title = info['title'].replace(" ", "_").replace("/", "_")
            return video_path, safe_title
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def load_raw_keypoints(npz_path):
    data = np.load(npz_path, allow_pickle=True)["data"]
    df = pd.DataFrame(data, columns=["frame","landmark","x","y","z","visibility"])
    pts_seq = []
    for f, group in df.groupby("frame"):
        group = group.sort_values("landmark")
        pts_seq.append(group[["x","y","z"]].values)
    return np.array(pts_seq)

def extract_3d_keypoints(video_path, dir_path, name):
    if video_path is None: return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_rows = []
    
    global_pos = np.array([0.0, 0.0, 0.0]) 
    prev_screen_hip = None 
    prev_hip_width = None
    prev_landmarks = None

    pbar = tqdm(total=total_frames, desc="Generating Subtle Motion", ascii=True)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_model.process(rgb)

        if res.pose_world_landmarks and res.pose_landmarks:
            world_lms = np.array([[lm.x, -lm.y, lm.z] for lm in res.pose_world_landmarks.landmark]) 
            screen_lms = np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark]) 
            
            # 스케일 계산
            nose = world_lms[0]
            hip_world = (world_lms[23] + world_lms[24]) / 2.0
            real_size = np.linalg.norm(nose - hip_world) 

            nose_screen = screen_lms[0]
            hip_screen = (screen_lms[23] + screen_lms[24]) / 2.0
            screen_size = np.linalg.norm(nose_screen - hip_screen) 

            scale_factor = real_size / (screen_size + 1e-6)

            curr_screen_hip = hip_screen
            curr_hip_width = np.linalg.norm(screen_lms[23] - screen_lms[24])

            if prev_screen_hip is not None:
                delta_screen = curr_screen_hip - prev_screen_hip
                
                # 컷 편집 제외
                if np.linalg.norm(delta_screen) < 0.2: 
                    # 이동량 계산 (0.2배 적용)
                    move_x = delta_screen[0] * scale_factor * MOVE_SENSITIVITY_X
                    
                    if prev_hip_width is not None:
                        width_diff = curr_hip_width - prev_hip_width
                        move_z = -width_diff * scale_factor * MOVE_SENSITIVITY_Z
                    else:
                        move_z = 0

                    global_pos[0] += move_x 
                    global_pos[2] += move_z 

            prev_screen_hip = curr_screen_hip
            prev_hip_width = curr_hip_width

            final_pose = world_lms.copy()
            final_pose[:, 0] += global_pos[0] 
            final_pose[:, 2] += global_pos[2] 
            
            for i in range(33):
                pose_rows.append({
                    "frame": frame_idx, "landmark": i,
                    "x": final_pose[i, 0], "y": final_pose[i, 1], "z": final_pose[i, 2],
                    "visibility": res.pose_world_landmarks.landmark[i].visibility
                })
            prev_landmarks = final_pose 

        else:
            if prev_landmarks is not None:
                for i in range(33):
                    pose_rows.append({
                        "frame": frame_idx, "landmark": i,
                        "x": prev_landmarks[i, 0], "y": prev_landmarks[i, 1], "z": prev_landmarks[i, 2],
                        "visibility": 0.0
                    })

        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()

    if not pose_rows: return

    df = pd.DataFrame(pose_rows)
    out_path = f"{dir_path}/{name}.npz"
    np.savez(out_path, data=df.to_numpy())
    
    base = os.path.basename(out_path).replace(".npz", "")
    raw_seq = load_raw_keypoints(out_path)
    np.save(os.path.join(RAW_TEST_DIR, f"{base}_raw.npy"), raw_seq)
    print(f"✅ Data Ready (Sensitivity 0.2): {base}_raw.npy")

def main():
    target_url = 'https://www.youtube.com/shorts/o_g_8WxGOrE' 
    video_path, name = download_youtube(target_url)
    if video_path:
        extract_3d_keypoints(video_path, RAW_TEST_DIR, name)

if __name__ == "__main__":
    main()