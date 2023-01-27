import cv2
import os
import numpy as np
import imutils

def extract_frame(video_path: str, output_folder: str, output_file: str, frame_idx: int = 0) -> str:
    input_video = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = input_video.read()
        frame = imutils.resize(frame, height=352)
        frame = cv2.copyMakeBorder(frame, left=295, right=296, top=0, bottom=0, borderType=cv2.BORDER_CONSTANT)
        if frame_idx == frame_count:
            path = os.path.join(output_folder, output_file % frame_idx)
            cv2.imwrite(path, frame)
            return output_file % frame_idx
        frame_count += 1

def load_depth_map(current_folder: str, max_depth: int = None):
    input_video = os.path.join(current_folder, 'video.mp4')
    depth_map_path = current_folder + (f'depth_map_{max_depth}.npy' if max_depth is not None else 'depth_map.npy')
    if not os.path.exists(depth_map_path):
        print('Depth map generation.')
        scaled_image_name = extract_frame(input_video, current_folder, 'frame_%d_scaled.jpg')
        print(f'Extracted scaled frame to {scaled_image_name}')
        from modules.depth_map.pixelformer.test import generate_depth_map
        generate_depth_map(current_folder, scaled_image_name, max_depth_o=max_depth)
    else:
        print('Depth map found.')
        
    if not os.path.exists(depth_map_path):
        raise Exception('Depth Map could not be generated.')

    with open(depth_map_path, 'rb') as f:
        our_meters = np.load(f)

    return our_meters
