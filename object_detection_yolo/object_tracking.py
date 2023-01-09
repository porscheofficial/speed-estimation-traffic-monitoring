from datetime import datetime
from importlib import reload
from multiprocessing import Pool
import cv2
import os
import configparser
from object_detection import ObjectDetection
import math
from get_fps import give_me_fps
import pandas as pd
import logging
import json
from object_detection_yolo.modules.evaluation.evaluate import plot_absolute_error
from paths import session_path
import copy
import imutils
import time
import uuid
from utils.object_tracking import Point, Car, clamp
from modules.depth_map.depth_map_utils import load_depth_map

config = configparser.ConfigParser()
config.read('object_detection_yolo/config.ini')

def run(data_dir, max_depth=None, fps=None):
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
    od = ObjectDetection()

    input_video = cv2.VideoCapture(path_to_video)

    fps = give_me_fps(path_to_video) if fps is None else fps

    sliding_window = 15 * fps
    text_color = (255,255,255)

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

    while True:
        ret, frame = input_video.read()
        frame = imutils.resize(frame, height=352)
        frame = cv2.copyMakeBorder(frame, left=295, right=296, top=0, bottom=0, borderType=cv2.BORDER_CONSTANT)
        #cv2.imwrite("object-detection-yolo/frames_detected/frame%d_new_scaled.jpg" % frame_count, frame)
        frame_count += 1
        if not ret:
            break
        # Point current frame
        center_points_cur_frame = []
        center_points_prev_frame = []

        # Detect objects on frame
        (class_ids, scores, boxes) = od.detect(frame)
        for box in boxes:
            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            if cy < depth_map.shape[0] and cx < depth_map.shape[1]:
                meters = depth_map[cy][cx]
            else:
                meters = 0
            center_points_cur_frame.append(Point(cx, cy, meters, x, y, w, h, frame_count))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

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
                total_movement = math.sqrt(math.pow(x_movement, 2) + math.pow(y_movement, 2))

                meters_moved = abs(tracking_objects[object_id].meters_moved - tracking_objects_prev[object_id].meters_moved)

                if object_id in cars:
                    cars[object_id].meters_moved += clamp(meters_moved, 0.0, 0.7)
                    cars[object_id].frames_seen += 1
                    cars[object_id].frame_end += 1                    
                else:
                    cars[object_id] = Car(meters_moved, 1, frame_count, frame_count)


            #cv2.circle(frame, (point.x, point.y), 5, (0, 0, 255), -1)
            if not meters_moved or meters_moved < 0.001:
                cv2.putText(frame, f"ID:{object_id}", (point.x + point.w + 5, point.y + point.h), 0, 1, (0,0,255), 2)
            else:
                cv2.putText(frame, f"ID:{object_id}", (point.x + point.w + 5, point.y + point.h), 0, 1, (0,255,0), 2)

        # Make a copy of the points
        center_points_prev_frame = copy.deepcopy(center_points_cur_frame)
        tracking_objects_prev = copy.deepcopy(tracking_objects)

        #key = cv2.waitKey(1)
        # if key == 27:
        #     break
        if frame_count >= fps and frame_count % (60 * fps) == 0:
            # every minute
            total_speed = 0
            car_count = 0
        
            for car_id, car in cars.items():

                if car.frame_end >= frame_count - 60 * fps and car.frames_seen > 5 and car.meters_moved > 0.05:
                    car_count += 1
                    total_speed += (car.meters_moved) / (car.frames_seen / fps)
            if car_count > 0:
                print(f"Average speed last minute: {(total_speed / car_count) * 3.6:.2f} km/h")
                avg_speed = round((total_speed / car_count) * 3.6, 2)
                logging.info(json.dumps(dict(frameId=frame_count, avgSpeedLastMinute=avg_speed)))

        
        if frame_count >= fps and frame_count % (5 * fps) == 0:
            # every 5 seconds
            total_speed = 0
            car_count = 0
        
            for car_id, car in cars.items():

                if car.frame_end >= frame_count - sliding_window and car.frames_seen > 5 and car.meters_moved > 0.05:
                    car_count += 1
                    total_speed += (car.meters_moved) / (car.frames_seen / fps)
            if car_count > 0:
                print(f"Average speed: {(total_speed / car_count) * 3.6:.2f} km/h")
                avg_speed = round((total_speed / car_count) * 3.6, 2)
                logging.info(json.dumps(dict(frameId=frame_count)))


        cv2.putText(
            frame,
            f"Timestamp: {(frame_count / fps):.2f} s",
            (7, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2,
        )

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (7, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2
        )

        try:
            cv2.putText(
                frame,
                f"Ground truth: {avg_speed_for_time((frame_count/fps)-sliding_window,frame_count/fps):.2f} km/h",
                (7, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color,
                2
            )
        except:
            cv2.putText(
                frame,
                f"Problems retrieving ground truth for frame: {frame_count}.",
                (7, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color,
                2
            )

        cv2.putText(
            frame,
            f"Estimated speed: {avg_speed} km/h",
            (7, 160),    
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2
        )
        if frame_count % 500 == 0:
            print(f"Frame no. {frame_count} time since start: {(time.time()-start):.2f}s")

        #cv2.imshow("Frame", frame)
        if frame_count % 3000 == 0:
            cv2.imwrite(f"object-detection-yolo/frames_detected/{run_id}_frame_{frame_count}.jpg", frame)

    input_video.release()
    cv2.destroyAllWindows()
    logging.shutdown()

    return log_name


if __name__ == '__main__':
    fps = int(config["DEFAULT"]["fps"])

    logs = []
    logs += [run(session_path, 85, fps)]
    logs += [run(session_path, 100, fps)]
    logs += [run(session_path, 120, fps)]

    ### Evaluation
    plot_absolute_error(logs, 'logs/')
