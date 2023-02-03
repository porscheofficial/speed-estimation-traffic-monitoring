import cv2
import os
import numpy as np
import imutils
from modules.depth_map.pixelformer.test import generate_depth_map


def get_padding_right(shape, height = 352):
    h,w = shape[:2]

    r = height / float(h)
    dim = (int(w * r), height)

    width = min(dim[0], 1216)

    return 1216 - width

def resize_input(frame):
    padding_right = get_padding_right(frame.shape)

    frame = imutils.resize(frame, height=352)

    if frame.shape[1] > 1216:
        frame = frame[:, 0:1215]

    frame = cv2.copyMakeBorder(frame, left=0, right=padding_right, top=0, bottom=0, borderType=cv2.BORDER_CONSTANT)

    return frame

def resize_output(prediction, shape):
    padding_right = get_padding_right(shape)

    prediction = prediction[:, 0:-padding_right]

    return cv2.resize(prediction, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def extract_frame(video_path: str, output_folder: str, output_file: str, frame_idx: int = 0) -> str:
    input_video = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = input_video.read()
        original_shape = frame.shape

        frame = resize_input(frame)

        if frame_idx == frame_count:
            path = os.path.join(output_folder, output_file % frame_idx)
            cv2.imwrite(path, frame)
            return output_file % frame_idx, original_shape
        frame_count += 1

def load_depth_map_from_file(current_folder: str, max_depth: int = None, frame: int = 0):
    input_video = os.path.join(current_folder, 'video.mp4')
    depth_map_path = current_folder + (f'depth_map_{max_depth}_{frame}.npy' if max_depth is not None else 'depth_map.npy')
    original_shape = (1080, 1920)
    
    if not os.path.exists(depth_map_path):
        print('Depth map generation.')
        scaled_image_name, original_shape = extract_frame(input_video, current_folder, 'frame_%d_scaled.jpg')
        print(f'Extracted scaled frame to {scaled_image_name}')
        return resize_output(generate_depth_map(current_folder, scaled_image_name, max_depth_o=max_depth), original_shape)
    else:
        print('Depth map found.')
        
    if not os.path.exists(depth_map_path):
        raise Exception('Depth Map could not be generated.')

def load_depth_map(current_folder: str, frame, max_depth=1):
    path = os.path.join(current_folder, 'frame_scaled.jpg')
    cv2.imwrite(path, frame)
    depth_map = generate_depth_map(current_folder, 'frame_scaled.jpg', max_depth_o=max_depth)
    return resize_output(depth_map, frame.shape)
