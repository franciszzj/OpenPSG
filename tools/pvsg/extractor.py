import os
from pathlib import Path
import cv2
from tqdm import tqdm
import os.path as osp


def makedir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

video_root = "./vidor/videos"
save_root = "./vidor/images"

video_root = Path(video_root)
videos = video_root.rglob("*.mp4")
for video in tqdm(videos):
    video_path = str(video)
    video_name = video_path.split("/")[-1].split(".")[0]
    save_dir = osp.join(save_root, video_name)
    makedir_if_not_exist(save_dir)
    cap = cv2.VideoCapture(video_path)  
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    count = 0
    while success and count < n_frames:
        success, image = cap.read()
        if success:
            cv2.imwrite(os.path.join(save_dir,"{:04d}.png".format(count)), image)
            count+=1
