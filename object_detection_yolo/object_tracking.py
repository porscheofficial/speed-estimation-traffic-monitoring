from datetime import datetime
from importlib import reload
from multiprocessing import Pool
import sys
import cv2
import os
import configparser
from modules.object_detection.yolov5.object_detection import (
    ObjectDetection as ObjectDetectionCustom,
)
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
from collections import defaultdict

config = configparser.ConfigParser()
config.read("object_detection_yolo/config.ini")


def calc_viewing_distance(car_lines):
    car_lines.sort(key=lambda x: x[0])

    box_count = 0
    start = None
    for box in car_lines:
        if start is None:
            start = box[0] + box[1]
            box_count += 1
            continue
        if box[0] >= start:
            start = box[0] + box[1]
            box_count += 1
    return box_count * 5

def run(
    data_dir, max_depth=None, fps=None, max_frames=None, custom_object_detection=False
):
    reload(logging)
    path_to_video = os.path.join(data_dir, "video.mp4")

    # Load Cars
    cars = pd.read_csv(data_dir + "cars.csv")

    def avg_speed_for_time(timeStart, timeEnd):
        cars_to_avg = cars.loc[cars["start"].gt(timeStart) & cars["end"].le(timeEnd)]
        return cars_to_avg["speed"].mean()

    run_id = uuid.uuid4().hex[:10]
    print(f"Run No.: {run_id}")

    # Initialize logging
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"logs/{now_str}_run_{run_id}.log"
    logging.basicConfig(
        filename=f"logs/{now_str}_run_{run_id}.log", level=logging.DEBUG
    )
    logging.info(
        f'Run No.: {run_id}, Video: {data_dir}, Max Depth: {"None" if max_depth is None else max_depth}'
    )

    # Initialize Object Detection
    start = time.time()

    if custom_object_detection:
        weights = "object_detection_yolo/model_weights/yolov5/best.pt"
        od = ObjectDetectionCustom(weights=weights)
    else:
        od = ObjectDetection()

    input_video = cv2.VideoCapture(path_to_video)

    fps = give_me_fps(path_to_video) if fps is None else fps

    sliding_window = 15 * fps
    text_color = (255, 255, 255)
    # text_color = (0,0,0)

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

    # meta_appr
    avg_frame_count = float(config.get("analyzer", "avg_frame_count"))
    speed_limit = int(config.get("analyzer", "speed_limit"))

    if custom_object_detection:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    car_lines = []

    depth_map = None

    while True:
        ret, frame = input_video.read()
        if not ret:
            break

        original_frame = frame

        if custom_object_detection:
            frame = fgbg.apply(frame)
            path_to_frame = f"object_detection_yolo/frames_detected/frame_{run_id}.jpg"
            cv2.imwrite(path_to_frame, frame)
        else:
            frame = imutils.resize(frame, height=352)
            frame = cv2.copyMakeBorder(
                frame,
                left=295,
                right=296,
                top=0,
                bottom=0,
                borderType=cv2.BORDER_CONSTANT,
            )
        # cv2.imwrite("object_detection_yolo/frames_detected/frame%d_new_scaled.jpg" % frame_count, frame)
        frame_count += 1

        # for shake_detection
        frames.append(original_frame)
        if len(frames) == 2 and frames[1] is not None:
            out = frames[0] - frames[1]
            # print(out)
            out[-11:11] = 0  # remove some random noise
            zeros = out.size - np.count_nonzero(out)
            size = out.size
            percentage_of_zeros = zeros / size
            full_changes.append(percentage_of_zeros)
            frames = []
            q1 = np.percentile(full_changes, 25)
            if len(full_changes) >= 100:
                starter_threshold = q1
            if (
                percentage_of_zeros < starter_threshold / 4
            ):  # divided by 4 for hard move
                cont = "HARD MOVE HAPPENED"
                cont_hard = cont + " at frame: " + str(frame_count)
                full_changes = []
                # print(cont_hard)
                logging.info(
                    f"Run No.: {run_id}, Video: {data_dir}, Hard Move Detected Frame: {frame_count}"
                )
            else:
                # print("no hard move detected")
                nothing = 42  # xd
        # this part of shake_detection averages the pixel changes for comparison
        if len(full_changes) == 400:
            avg_changes = sum(full_changes) / len(full_changes)
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

            if depth_map is not None and (cy < depth_map.shape[0] and cx < depth_map.shape[1]):
                meters = depth_map[cy][cx]
            else:
                meters = 0

            if custom_object_detection:
                try:
                    cropped_frame = frame[y:y+h, x:x+w]
                    car_length_in_pixels = get_pixel_length_of_car(cropped_frame)
                    avg_car_in_meters = 5
                    if car_length_in_pixels is None:
                        ppm = None
                    else:
                        ppm = car_length_in_pixels/avg_car_in_meters
                except:
                    ppm = None
            else:
                ppm = None
            center_points_cur_frame.append(
                Point(cx, cy, meters, x, y, w, h, frame_count, ppm)
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Only at the beginning we compare previous and current frame
        if frame_count <= 2:
            center_points_prev_frame_copy = center_points_prev_frame.copy()
            for point in center_points_cur_frame:
                for point2 in center_points_prev_frame_copy:
                    distance = math.hypot(point2.x - point.x, point2.y - point.y)

                    if distance < 20:
                        point.object_id = track_id
                        tracking_objects[track_id] = point
                        tracking_objects_diff_map[track_id] = (
                            point2.x - point.x,
                            point2.y - point.y,
                        )
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

                # Remove IDs lost
                if not object_exists:
                    
                    # for idx in range(1, len(car_lines[object_id])):
                    #     idx_prev = idx - 1
                    #     frame = cv2.line(frame, car_lines[object_id][idx_prev], car_lines[object_id][idx], (255,255,255), 8)

                    tracking_objects.pop(object_id)

            # Add new IDs found
            for point in center_points_cur_frame:
                tracking_objects[track_id] = point
                track_id += 1

        if depth_map is None and len(car_lines) >= 200:
            # Load depth map
            max_depth = calc_viewing_distance(car_lines)
            depth_map = load_depth_map(data_dir, max_depth=max_depth)


        for object_id, point in tracking_objects.items():
            meters_moved = None
            if object_id in tracking_objects_prev:
                x_movement = (
                    tracking_objects[object_id].x - tracking_objects_prev[object_id].x
                )
                y_movement = (
                    tracking_objects[object_id].y - tracking_objects_prev[object_id].y
                )

                # car direction
                # print(object_id, x_movement, y_movement)
                direction_indicator = 0
                if y_movement < 0:
                    direction = "away from cam"
                    direction_indicator += 1
                if y_movement == 0:
                    direction = "no y movement"
                    direction_indicator += 0
                if y_movement > 0:
                    direction = "towards cam"
                    direction_indicator -= 1

                total_movement = math.sqrt(
                    math.pow(x_movement, 2) + math.pow(y_movement, 2)
                )

                if custom_object_detection:
                    if (
                        tracking_objects_prev[object_id].ppm
                        and tracking_objects[object_id].ppm
                    ):
                        avg_ppm = (
                            tracking_objects_prev[object_id].ppm
                            + tracking_objects[object_id].ppm
                        ) / 2
                    elif (
                        tracking_objects_prev[object_id].ppm
                        or tracking_objects[object_id].ppm
                    ):
                        avg_ppm = (
                            tracking_objects_prev[object_id].ppm
                            or tracking_objects[object_id].ppm
                        )
                    else:
                        continue

                    meters_moved = total_movement / avg_ppm
                else:
                    meters_moved = abs(
                        tracking_objects[object_id].meters_moved
                        - tracking_objects_prev[object_id].meters_moved
                    )

                if object_id in cars:
                    cars[object_id].meters_moved += clamp(meters_moved, 0.0, 0.7)
                    cars[object_id].frames_seen += 1
                    cars[object_id].frame_end += 1
                else:
                    cars[object_id] = Car(meters_moved, 1, frame_count, frame_count)

            # if object_id in tracking_objects_prev and object_id in tracking_objects:
            #     car_lines[object_id].append((tracking_objects[object_id].x, tracking_objects[object_id].w))
            car_lines.append((tracking_objects[object_id].x, tracking_objects[object_id].w))

            #cv2.circle(frame, (point.x, point.y), 5, (0, 0, 255), -1)
            if not meters_moved or meters_moved < 0.001:
                cv2.putText(frame, f"ID:{object_id}", (point.x + point.w + 5, point.y + point.h), 0, 1, (255,255,255), 2)
            else:
                cv2.putText(frame, f"ID:{object_id}", (point.x + point.w + 5, point.y + point.h), 0, 1, (255,255,255), 2)

        # Make a copy of the points
        center_points_prev_frame = copy.deepcopy(center_points_cur_frame)
        tracking_objects_prev = copy.deepcopy(tracking_objects)

        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
        if frame_count >= fps and frame_count % (60 * fps) == 0:
            # every minute
            total_speed = 0
            car_count = 0
            total_speed_meta_appr = 0

            for car_id, car in cars.items():

                if (
                    car.frame_end >= frame_count - 60 * fps
                    and car.frames_seen > 5
                    and car.meters_moved > 0.05
                ):
                    car_count += 1
                    total_speed += (car.meters_moved) / (car.frames_seen / fps)
                    total_speed_meta_appr += (
                        avg_frame_count / int(car.frames_seen)
                    ) * speed_limit

            if car_count > 0:
                print(
                    f"Average speed last minute: {(total_speed / car_count) * 3.6:.2f} km/h"
                )
                print(
                    f"Average META speed last minute: {(total_speed_meta_appr / car_count)} km/h"
                )

                avg_speed = round((total_speed / car_count) * 3.6, 2)
                logging.info(
                    json.dumps(dict(frameId=frame_count, avgSpeedLastMinute=avg_speed))
                )

        if frame_count >= fps and frame_count % (5 * fps) == 0:
            # every 5 seconds
            total_speed = 0
            car_count = 0
            total_speed_meta_appr = 0

            for car_id, car in cars.items():

                if (
                    car.frame_end >= frame_count - sliding_window
                    and car.frames_seen > 5
                    and car.meters_moved > 0.05
                ):
                    car_count += 1
                    total_speed += (car.meters_moved) / (car.frames_seen / fps)
                    total_speed_meta_appr += (
                        avg_frame_count / int(car.frames_seen)
                    ) * speed_limit

            if car_count > 0:
                print(f"Average speed: {(total_speed / car_count) * 3.6:.2f} km/h")
                print(f"Average META speed: {(total_speed_meta_appr / car_count)} km/h")

                avg_speed = round((total_speed / car_count) * 3.6, 2)
                logging.info(
                    json.dumps(
                        dict(frameId=frame_count, avgSpeedLast15Seconds=avg_speed)
                    )
                )

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
            frame, f"FPS: {fps}", (7, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2
        )

        # try:
        #     cv2.putText(
        #         frame,
        #         f"Ground truth: {avg_speed_for_time((frame_count/fps)-sliding_window,frame_count/fps):.2f} km/h",
        #         (7, 130),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         text_color,
        #         2
        #     )
        # except:
        #     cv2.putText(
        #         frame,
        #         f"Problems retrieving ground truth for frame: {frame_count}.",
        #         (7, 130),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         text_color,
        #         2
        #     )

        # cv2.putText(
        #     frame,
        #     f"Estimated speed: {avg_speed} km/h",
        #     (7, 160),    
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     text_color,
        #     2
        # )
        if frame_count % 500 == 0:
            print(
                f"Frame no. {frame_count} time since start: {(time.time()-start):.2f}s"
            )

        #cv2.imwrite(f"object_detection_yolo/frames_detected/frame_after_detection.jpg", frame)

        #cv2.imshow("Frame", frame)
        # if frame_count % 3000 == 0:
        #     cv2.imwrite(f"object_detection_yolo/frames_detected/{run_id}_frame_{frame_count}.jpg", frame)

        if max_frames is not None and frame_count >= max_frames:
            break

    input_video.release()
    cv2.destroyAllWindows()
    logging.shutdown()

    return log_name


def main():
    fps = config.getint("DEFAULT", "fps")
    custom_object_detection = config.getboolean("DEFAULT", "custom_object_detection")

    max_frames = fps * 60 * 15  # fps * sec * min
    max_depth_tests = [100]
    session_path_local = sys.argv[1] if len(sys.argv) > 1 else session_path

    logs = []
    for max_depth in max_depth_tests:
        logs += [
            run(
                session_path_local,
                max_depth,
                fps,
                max_frames=max_frames,
                custom_object_detection=custom_object_detection,
            )
        ]

    ### Evaluation
    plot_absolute_error(logs, "logs/")


if __name__ == "__main__":
    main()
