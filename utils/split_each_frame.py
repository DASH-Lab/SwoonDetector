import cv2
import glob
import os
import imageio
# from PIL import Image
VIDEO_PATH = glob.glob('/media/data2/AGC_system_test/VIDEO_RAW_NIGHT/*.MP4')
TO_SAVE_PATH = '/media/data2/AGC_system_test/validation_data_night'
os.makedirs(TO_SAVE_PATH, exist_ok=True)
for video in VIDEO_PATH:
    frames = imageio.mimread(video, memtest=False)
    filename_only = video.split("/")[-1].split(".")[0]
    ret = True
    print(video)
    count = 0
    save_dir = os.path.join(TO_SAVE_PATH, filename_only)
    os.makedirs(save_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        cv2.imwrite(os.path.join(save_dir, filename_only+" "+str(idx).zfill(3)+".jpg"), frame[:,:,::-1])
