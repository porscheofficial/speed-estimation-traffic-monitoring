# can get shortened and more lightweight even more if we clean up more; atm like this with lot of unnece.
#  stuff because of dependencies for beeing able to run it

from datetime import datetime
from importlib import reload
from multiprocessing import Pool
import sys
import cv2
import os
import configparser
from modules.object_detection.yolov5.object_detection import ObjectDetection as ObjectDetectionCustom
from modules.object_detection.yolov4.object_detection import ObjectDetection
import math
from get_fps import give_me_fps
import pandas as pd
import logging
import json
from paths import session_path
import copy
import imutils
import time
import uuid
from utils.object_tracking import Point, Car, clamp
from modules.evaluation.evaluate import plot_absolute_error
from modules.depth_map.depth_map_utils import load_depth_map
import numpy as np
from modules.regression.regression_calc import get_pixel_length_of_car
from analyze_logs import analyzer


config = configparser.ConfigParser()
config.read('object_detection_yolo/config.ini')

#set for callibr:
max_cars = 40
callibr = []

def run(data_dir, max_depth=None, fps=None, max_frames=None, custom_object_detection=False):
    reload(logging)
    path_to_video = os.path.join(data_dir, 'video.mp4')

    # Load Cars
    cars = pd.read_csv(data_dir + "cars.csv")

    def avg_speed_for_time(timeStart, timeEnd):
        cars_to_avg = cars.loc[cars['start'].gt(timeStart) & cars['end'].le(timeEnd)]
        return cars_to_avg['speed'].mean()

    run_id = uuid.uuid4().hex[:10]
    print(f"Run No.: {run_id}")

    # Initialize logging
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f'logs/{now_str}_run_{run_id}.log'
    logging.basicConfig(filename=f'logs/{now_str}_run_{run_id}.log', level=logging.DEBUG)
    logging.info(f'Run No.: {run_id}, Video: {data_dir}, Max Depth: {"None" if max_depth is None else max_depth}')

    # Load depth map
    depth_map = load_depth_map(data_dir, max_depth=max_depth)

    # Initialize Object Detection
    start = time.time()

    if custom_object_detection:
        source = "object_detection_yolo/frames_detected/frame.jpg"
        weights = "object_detection_yolo/model_weights/yolov5/best.pt"
        od = ObjectDetectionCustom(weights=weights, source=source)
    else:
        od = ObjectDetection()

    input_video = cv2.VideoCapture(path_to_video)

    fps = give_me_fps(path_to_video) if fps is None else fps

    sliding_window = 15 * fps
    text_color = (255,255,255)
    #text_color = (0,0,0)

    # Initialize count
    frame_count = 0
    tracking_objects = {}

    # TODO: Remove if unused
    tolerance_px = 3
    tracking_objects_diff_map = {}

    tracking_objects_prev = {}
    cars = {}
    track_id = 0
    avg_speed = "calculating"

    # for shake_detection
    frames = []
    full_changes = []
    starter_threshold = 0.4
    # for shake_detection

    if custom_object_detection:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 

    while True:
        ret, frame = input_video.read()
        if not ret:
            break

        original_frame = frame

        if custom_object_detection:
            frame = fgbg.apply(frame)
            path_to_frame = "object_detection_yolo/frames_detected/frame.jpg"
            cv2.imwrite(path_to_frame, frame)
        else:
            frame = imutils.resize(frame, height=352)
            frame = cv2.copyMakeBorder(frame, left=295, right=296, top=0, bottom=0, borderType=cv2.BORDER_CONSTANT)
        #cv2.imwrite("object_detection_yolo/frames_detected/frame%d_new_scaled.jpg" % frame_count, frame)
        frame_count += 1

        # for shake_detection
        frames.append(original_frame)
        if len(frames) == 2 and frames[1] is not None:
            out = frames[0] - frames[1]
            # print(out)
            out[-11:11] = 0  # remove some random noise
            zeros = out.size - np.count_nonzero(out)
            size = out.size
            percentage_of_zeros = zeros/size
            full_changes.append(percentage_of_zeros)
            frames = []
            q1 = np.percentile(full_changes, 25)
            if len(full_changes) >= 100:
                starter_threshold = q1
            if percentage_of_zeros < starter_threshold/4:  # divided by 4 for hard move
                cont = "HARD MOVE HAPPENED"
                cont_hard = cont + " at frame: " + str(frame_count)
                full_changes = []
                # print(cont_hard)
                logging.info(
                    f'Run No.: {run_id}, Video: {data_dir}, Hard Move Detected Frame: {frame_count}')
            else:
                # print("no hard move detected")
                nothing = 42 # xd
        # this part of shake_detection averages the pixel changes for comparison
        if len(full_changes) == 400:
            avg_changes = sum(full_changes)/len(full_changes)
            q1 = np.percentile(full_changes, 25)
            # print("1Q")
            # print(q1)
            # plt.hist(full_changes)
            # plt.show()
            length_f = len(full_changes)
            last100 = length_f - 102
            del full_changes[last100:]
            # end shake_detection

        
        # Point current frame
        center_points_cur_frame = []
        center_points_prev_frame = []

        # Detect objects on frame
        if custom_object_detection:
            boxes = od.detect(path_to_frame)
            if len(boxes) == 0:
                continue

        else:
            (class_ids, scores, boxes) = od.detect(frame)
        for box in boxes:
            (x, y, w, h) = box.astype(int)
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            if cy < depth_map.shape[0] and cx < depth_map.shape[1]:
                meters = depth_map[cy][cx]
            else:
                meters = 0

            if custom_object_detection:
                cropped_frame = frame[y:y+h, x:x+w]
                car_length_in_pixels = get_pixel_length_of_car(cropped_frame)
                avg_car_in_meters = 5
                if car_length_in_pixels is None:
                    ppm = None
                else:
                    ppm = car_length_in_pixels/avg_car_in_meters
            else:
                ppm = None
            center_points_cur_frame.append(Point(cx, cy, meters, x, y, w, h, frame_count, ppm))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)

        # Only at the beginning we compare previous and current frame
        if frame_count <= 2:
            center_points_prev_frame_copy = center_points_prev_frame.copy()
            for point in center_points_cur_frame:
                for point2 in center_points_prev_frame_copy:
                    distance = math.hypot(point2.x - point.x, point2.y - point.y)

                    if distance < 20:
                        tracking_objects[track_id] = point
                        tracking_objects_diff_map[track_id] = (point2.x - point.x, point2.y - point.y)
                        track_id += 1
                        center_points_prev_frame_copy.remove(point2)
                        break
        else:
            tracking_objects_copy = tracking_objects.copy()
            center_points_cur_frame_copy = center_points_cur_frame.copy()

            for object_id, point2 in tracking_objects_copy.items():
                object_exists = False
                for point in center_points_cur_frame_copy:
                    distance = math.hypot(point2.x - point.x, point2.y - point.y)

                    # Update IDs position
                    if distance < 50:
                        tracking_objects[object_id] = point
                        object_exists = True
                        if point in center_points_cur_frame:
                            center_points_cur_frame.remove(point)
                            # Point should not match to multiple boxes
                            # TODO: Only take closest!
                            center_points_cur_frame_copy.remove(point)
                        break
                        continue

                # Remove IDs lost
                if not object_exists:
                    tracking_objects.pop(object_id)

            # Add new IDs found
            for point in center_points_cur_frame:
                tracking_objects[track_id] = point
                track_id += 1

        for object_id, point in tracking_objects.items():
            meters_moved = None
            if object_id in tracking_objects_prev:
                x_movement = tracking_objects[object_id].x - tracking_objects_prev[object_id].x
                y_movement = tracking_objects[object_id].y - tracking_objects_prev[object_id].y

                # car direction
                # print(object_id, x_movement, y_movement)
                direction_indicator = 0
                if y_movement < 0:
                    direction = 'away from cam'
                    direction_indicator+=1
                if y_movement == 0:
                    direction = 'no y movement'
                    direction_indicator+=0
                if y_movement > 0:
                    direction = 'towards cam'
                    direction_indicator -=1

                # print(“car: ” + str(object_id), direction, side, frame_count)
                #callibr.append([object_id, direction, frame_count])
                dictionary = {
                                'car_id': object_id,
                                'direction_indicator': direction_indicator,
                                'frame_count': frame_count
                            }
                # print(frame_count)
                logging.info(json.dumps(dictionary))

                # stop at max cars
                if object_id >= max_cars:
                    input_video.release()
                    cv2.destroyAllWindows()
                    logging.shutdown()
                    break

        # Make a copy of the points
        center_points_prev_frame = copy.deepcopy(center_points_cur_frame)
        tracking_objects_prev = copy.deepcopy(tracking_objects)

        if max_frames is not None and frame_count >= max_frames:
            break

    input_video.release()
    cv2.destroyAllWindows()
    logging.shutdown()

    print("now analyzing...")
    analyzer(log_file=log_name)

    return log_name

def main():
    fps = config.getint("DEFAULT", "fps")
    custom_object_detection = config.getboolean("DEFAULT", "custom_object_detection")

    max_frames = fps * 60 * 15 # fps * sec * min
    max_depth_tests = [100]
    session_path_local = sys.argv[1] if len(sys.argv) > 1 else session_path 

    logs = []
    for max_depth in max_depth_tests:
        logs += [run(session_path_local, max_depth, fps, max_frames=max_frames, custom_object_detection=custom_object_detection)]

    ### Evaluation
    plot_absolute_error(logs, 'logs/')

if __name__ == '__main__':
    main()