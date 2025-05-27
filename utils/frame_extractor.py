# Converst video to frames
# Extracts frmaes from original HockeyFights .avi videos and stores them in folders training.

import os
import cv2
from pathlib import Path


def extract_frames_from_videos(video_dir, output_dir, label_name, frame_rate=1):
    video_paths = list(Path(video_dir).glob("*.avi")) + \
        list(Path(video_dir).glob("*.mp4"))
    print(f"Found {len(video_paths)} videos in {video_dir}")

    for i, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(str(video_path))
        video_name = video_path.stem
        save_dir = os.path.join(output_dir, label_name, f"{label_name}_{i}")
        os.makedirs(save_dir, exist_ok=True)

        frame_count = 0
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_rate == 0:
                frame_path = os.path.join(
                    save_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            frame_index += 1
        cap.release()
        print(f"[{label_name.upper()}] {video_name}: {frame_count} frames extracted")
