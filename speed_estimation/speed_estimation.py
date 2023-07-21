"""Speed Estimation Pipeline.

This module defines the pipeline that takes a video or stream as input and estimates the speed of
the vehicles in the video footage.
Therefore, the different modules implemented in `speed_estimation/modules` are combined to derive
the vehicles' speeds.

The  main steps are:
1. Initialize the logging that will later on capture the speed estimates

2. Initialize the object detection that detects the vehicles. Per default a YoloV4 model.

3. Analyze if the video perspective changed. If yes pause the speed estimation and recalibrate
(future work)

4. Detect the vehicles and assign unique ids to each recognized bounding box. The ids are used for
tracking the vehicles and distinguish them.

5. As long as the pipeline is not calibrated, do a scaling factor estimation.

6. As soon as the calibration is done, do the speed estimation based on the scaling factor and the
detected bounding boxes for the vehicles.
"""
import argparse
import configparser
import json
import logging
import math
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from importlib import reload
from typing import Dict, List

import cv2
import torch
from tqdm import tqdm

from get_fps import get_fps
from modules import ObjectDetectionYoloV4
from modules import ShakeDetection
from modules.depth_map.depth_map_utils import DepthModel
from modules.evaluation.evaluate import plot_absolute_error
from modules.scaling_factor.scaling_factor_extraction import (
    GeometricModel,
    CameraPoint,
    get_ground_truth_events,
    offline_scaling_factor_estimation_from_least_squares,
)
from paths import SESSION_PATH, VIDEO_NAME
from utils.speed_estimation import (
    Direction,
    TrackingBox,
    Car,
    calculate_car_direction,
)

config = configparser.ConfigParser()
config.read("config.ini")


MAX_TRACKING_MATCH_DISTANCE = config.getint("tracker", "max_match_distance")
CAR_CLASS_ID = config.getint("tracker", "car_class_id")
NUM_TRACKED_CARS = config.getint("calibration", "num_tracked_cars")
NUM_GT_EVENTS = config.getint("calibration", "num_gt_events")
AVG_FRAME_COUNT = config.getfloat("analyzer", "avg_frame_count")
SPEED_LIMIT = config.getint("analyzer", "speed_limit")
SLIDING_WINDOW_SEC = config.getint("main", "sliding_window_sec")
FPS = config.getint("main", "fps")
CUSTOM_OBJECT_DETECTION = config.getboolean("main", "custom_object_detection")
OBJECT_DETECTION_MIN_CONFIDENCE_SCORE = config.getfloat(
    "main", "object_detection_min_confidence_score"
)


def run(
    path_to_video: str,
    data_dir: str,
    fps: int = 0,
    max_frames: int = 0,
    custom_object_detection: bool = False,
    enable_visual: bool = False
) -> str:
    """Run the full speed estimation pipeline.

    This method runs the full speed estimation pipeline, including the automatic calibration using
    depth maps, object detection, and speed estimation.

    @param path_to_video:
        The path to the video that should be analyzed. The default path is defined in
        `speed_estimation/paths.py`.

    @param data_dir:
        The path to the dataset directory. The default path is defined in
        `speed_estimation/paths.py`.

    @param fps:
        The frames per second of the video that should be analyzed. If nothing is defined, the
        pipeline will derive the fps automatically.

    @param max_frames:
        The maximum frames that should be analyzed. The pipeline will stop as soon as the given
        number is reached.

    @param custom_object_detection:
        If a custom/other object detection should be used, set this parameter to true. If the
        parameter is set to true, the pipeline expects the detection model in
        `speed_estimation/modules/custom_object_detection`. The default detection is a YoloV4 model.

    @param enable_visual:
        Enable a visual output of the detected and tracked cars. If the flag is disabled the frame
        speed_estimation/frames_detected/frame_after_detection.jpg will be updated.

    @return:
        The string to the log file containing the speed estimates.
    """
    reload(logging)

    run_id = uuid.uuid4().hex[:10]
    print(f"Run No.: {run_id}")

    # Initialize logging
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"logs/{now_str}_run_{run_id}.log"
    os.makedirs(os.path.dirname(log_name), exist_ok=True)

    logging.basicConfig(
        filename=f"logs/{now_str}_run_{run_id}.log", level=logging.DEBUG
    )
    logging.info("Run No.: %s, Video: %s", str(run_id), str(data_dir))

    start = time.time()

    # Initialize Object Detection
    if custom_object_detection:
        # Insert your custom object detection here
        object_detection = ObjectDetectionYoloV4()
    else:
        object_detection = ObjectDetectionYoloV4()

    input_video = cv2.VideoCapture(path_to_video)

    fps = get_fps(path_to_video) if fps == 0 else fps

    sliding_window = SLIDING_WINDOW_SEC * fps

    # Initialize running variables
    frame_count = 0
    track_id = 0
    tracking_objects: Dict[int, TrackingBox] = {}
    tracked_cars: Dict[int, Car] = {}
    tracked_boxes: Dict[int, List[TrackingBox]] = defaultdict(list)
    depth_model = DepthModel(data_dir, path_to_video)
    geo_model = GeometricModel(depth_model)
    is_calibrated = False
    text_color = (255, 255, 255)

    # for shake_detection
    shake_detection = ShakeDetection()

    progress_bar = tqdm(total=NUM_TRACKED_CARS)
    progress_bar.set_description("Calibrating")

    while True:
        ############################
        # load frame, shake detection and object detection
        ############################
        ret, frame = input_video.read()

        if frame_count == 0:
            # set normalization axes once at beginning
            center_x = int(frame.shape[1] / 2)
            center_y = int(frame.shape[0] / 2)
            geo_model.set_normalization_axes(center_x, center_y)

        if not ret:
            break

        # for shake_detection
        if shake_detection.is_hard_move(frame):
            logging.info(
                "Run No.: %s, Video: %s, Hard Move Detected Frame: %d",
                str(run_id),
                str(data_dir),
                frame_count,
            )

        ############################
        # Detect cars on frame
        ############################
        if custom_object_detection:
            # Detect cars with your custom object detection
            boxes = []
        else:
            (class_ids, scores, boxes) = object_detection.detect(frame)

            boxes = [
                boxes[i]
                for i, class_id in enumerate(class_ids)
                if class_id == CAR_CLASS_ID
                and scores[i] >= OBJECT_DETECTION_MIN_CONFIDENCE_SCORE
            ]

        # collect tracking boxes
        tracking_boxes_cur_frame: List[TrackingBox] = []
        for box in boxes:
            (x_coord, y_coord, width, height) = box.astype(int)
            center_x = int((x_coord + x_coord + width) / 2)
            center_y = int((y_coord + y_coord + height) / 2)

            tracking_boxes_cur_frame.append(
                TrackingBox(
                    center_x, center_y, x_coord, y_coord, width, height, frame_count
                )
            )

            cv2.rectangle(
                frame,
                (x_coord, y_coord),
                (x_coord + width, y_coord + height),
                (255, 0, 0),
                2,
            )

        ############################
        # assign tracking box IDs
        ############################
        for object_id, tracking_box_prev in tracking_objects.copy().items():
            min_distance = math.inf
            min_track_box = None

            # Find nearest bounding box
            for tracking_box_cur in tracking_boxes_cur_frame:
                distance = math.hypot(
                    tracking_box_prev.x_coord - tracking_box_cur.x_coord,
                    tracking_box_prev.y_coord - tracking_box_cur.y_coord,
                )

                # Only take bounding box if it is closest AND somewhat close to bounding
                # box (closer than MAX_TRACKING_...)
                if distance < min_distance and distance < MAX_TRACKING_MATCH_DISTANCE:
                    min_distance = distance
                    min_track_box = tracking_box_cur

            if min_track_box is not None:
                # Update tracking box for object if close box found
                tracking_objects[object_id] = min_track_box
                tracking_boxes_cur_frame.remove(min_track_box)
            else:
                # Remove IDs lost
                tracking_objects.pop(object_id)

        # Add new IDs found
        for tracking_box_cur in tracking_boxes_cur_frame:
            tracking_objects[track_id] = tracking_box_cur
            track_id += 1

        ############################
        # scaling factor estimation
        ############################
        if not is_calibrated:
            if len(tracked_boxes) >= NUM_TRACKED_CARS:
                # more than x cars were tracked
                ground_truth_events = get_ground_truth_events(tracked_boxes)
                print("Number of GT events: ", len(ground_truth_events))
                if len(ground_truth_events) >= NUM_GT_EVENTS:
                    # could extract more than x ground truth events
                    geo_model.scale_factor = 2 * (
                        offline_scaling_factor_estimation_from_least_squares(
                            geo_model, ground_truth_events
                        )
                    )
                    logging.info(
                        "Is calibrated: scale_factor: %d", geo_model.scale_factor
                    )
                    print(
                        f"Is calibrated: scale_factor: {geo_model.scale_factor}",
                        flush=True,
                    )
                    is_calibrated = True
                    progress_bar.close()
                    torch.cuda.empty_cache()
                    object_detection = ObjectDetectionYoloV4()

            progress_bar.update(len(tracked_boxes) - progress_bar.n)
            for object_id, tracking_box in tracking_objects.items():
                tracked_boxes[object_id].append(tracking_box)
        else:
            ############################
            # track cars
            ############################
            for object_id, tracking_box in tracking_objects.items():
                cv2.putText(
                    frame,
                    f"ID:{object_id}",
                    (
                        tracking_box.x_coord + tracking_box.width + 5,
                        tracking_box.y_coord + tracking_box.height,
                    ),
                    0,
                    1,
                    (255, 255, 255),
                    2,
                )
                if object_id in tracked_cars:
                    tracked_cars[object_id].tracked_boxes.append(tracking_box)
                    tracked_cars[object_id].frames_seen += 1
                    tracked_cars[object_id].frame_end += 1
                else:
                    tracked_cars[object_id] = Car(
                        [tracking_box], 1, frame_count, frame_count
                    )

            ############################
            # speed estimation
            ############################
            if frame_count >= fps and frame_count % (15 * fps) == 0:
                # every x seconds
                car_count_towards = 0
                car_count_away = 0
                total_speed_towards = 0
                total_speed_away = 0
                total_speed_meta_appr_towards = 0.0
                total_speed_meta_appr_away = 0.0
                ids_to_drop = []

                for car_id, car in tracked_cars.items():
                    if car.frame_end >= frame_count - sliding_window:
                        if 5 < car.frames_seen < 750:
                            car.direction = calculate_car_direction(car)
                            car_first_box = car.tracked_boxes[0]
                            car_last_box = car.tracked_boxes[-1]
                            meters_moved = geo_model.get_distance_from_camera_points(
                                CameraPoint(
                                    car_first_box.frame_count,
                                    car_first_box.center_x,
                                    car_first_box.center_y,
                                ),
                                CameraPoint(
                                    car_last_box.frame_count,
                                    car_last_box.center_x,
                                    car_last_box.center_y,
                                ),
                            )
                            if meters_moved <= 6:
                                continue

                            if car.direction == Direction.TOWARDS:
                                car_count_towards += 1
                                total_speed_towards += (meters_moved) / (
                                    car.frames_seen / fps
                                )
                                total_speed_meta_appr_towards += (
                                    AVG_FRAME_COUNT / int(car.frames_seen)
                                ) * SPEED_LIMIT
                            else:
                                car_count_away += 1
                                total_speed_away += (meters_moved) / (
                                    car.frames_seen / fps
                                )
                                total_speed_meta_appr_away += (
                                    AVG_FRAME_COUNT / int(car.frames_seen)
                                ) * SPEED_LIMIT

                    else:
                        # car is too old, drop from tracked_cars
                        ids_to_drop.append(car_id)

                for car_id in ids_to_drop:
                    del tracked_cars[car_id]

                if car_count_towards > 0:
                    avg_speed = round(
                        (total_speed_towards / car_count_towards) * 3.6, 2
                    )
                    print(f"Average speed towards: {avg_speed} km/h")
                    print(
                        f"Average META speed towards: "
                        f"{(total_speed_meta_appr_towards / car_count_towards)} km/h"
                    )
                    logging.info(
                        json.dumps(dict(frameId=frame_count, avgSpeedTowards=avg_speed))
                    )

                if car_count_away > 0:
                    avg_speed = round((total_speed_away / car_count_away) * 3.6, 2)
                    print(f"Average speed away: {avg_speed} km/h")
                    print(
                        f"Average META speed away: "
                        f"{(total_speed_meta_appr_away / car_count_away)} km/h"
                    )
                    logging.info(
                        json.dumps(dict(frameId=frame_count, avgSpeedAway=avg_speed))
                    )

        ############################
        # output text on video stream
        ############################
        timestamp = frame_count / fps
        cv2.putText(
            frame,
            f"Timestamp: {timestamp :.2f} s",
            (7, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2,
        )
        cv2.putText(
            frame, f"FPS: {fps}", (7, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2
        )

        if enable_visual:
            cv2.imshow("farsec", frame)
            cv2.waitKey(1000)
        else:
            cv2.imwrite("frames_detected/frame_after_detection.jpg", frame)

        if frame_count % 500 == 0:
            print(
                f"Frame no. {frame_count} time since start: {(time.time() - start):.2f}s"
            )
        frame_count += 1
        if max_frames != 0 and frame_count >= max_frames:
            if not is_calibrated:
                log_name = ""
            break

    input_video.release()
    cv2.destroyAllWindows()
    logging.shutdown()
    return log_name


def main(session_path_local: str, path_to_video: str, enable_visual: bool):
    """Run the speed estimation pipeline."""
    max_frames = FPS * 60 * 20  # fps * sec * min

    print(session_path_local)
    print(path_to_video)

    log_name = run(
        path_to_video,
        session_path_local,
        FPS,
        max_frames=max_frames,
        custom_object_detection=CUSTOM_OBJECT_DETECTION,
        enable_visual=enable_visual
    )

    if log_name is None:
        print("Calibration did not finish, skip evaluation.")
    else:
        # Evaluation
        plot_absolute_error([log_name], "logs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "session_path_local",
        nargs="?",
        help="Path to session (e.g., the directory where the video is stored)",
        default=SESSION_PATH,
    )
    parser.add_argument(
        "path_to_video",
        nargs="?",
        help="Path to video",
        default=os.path.join(SESSION_PATH, VIDEO_NAME),
    )
    parser.add_argument(
        "enable_visual",
        nargs="?",
        help="Enable visual output.",
        default=False,
    )
    args = parser.parse_args()

    # Run pipeline
    main(args.session_path_local, args.path_to_video, args.enable_visual)
