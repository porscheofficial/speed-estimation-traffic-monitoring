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
from modules.shake_detection.shake_detection import ShakeDetection
from paths import session_path
import copy
import imutils
import time
import uuid
from utils.object_tracking import Direction, Line, Point, TrackingBox, Car, clamp
from modules.evaluation.evaluate import plot_absolute_error
from modules.depth_map.depth_map_utils import load_depth_map, load_depth_map_from_file
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean
from modules.scaling_factor.scaling_factor_extraction import GroundTruthEvent, offline_scaling_factor_estimation_from_least_squares, GeometricModel, CameraPoint

config = configparser.ConfigParser()
config.read("object_detection_yolo/config.ini")

# ground_truth_events = [
#     GroundTruthEvent((frame_count, 296, 167), (frame_count, 1142, 967), 28.),
#     GroundTruthEvent((frame_count, 114, 190), (frame_count, 575, 828), 25.),
#     GroundTruthEvent((frame_count, 1141, 712), (frame_count, 1446, 973), 4.),
# ]

# Ihre Markierung 28m 
# Strassenpfosen 25m 
# Auto 4m 
# geo_model.scale_factor = 84
# ground_truth_sanity_check = dict(
#     marked_points=geo_model.get_distance_from_camera_points(CameraPoint(frame_count, 296, 167), CameraPoint(frame_count, 1142, 967)),
#     delineator=geo_model.get_distance_from_camera_points(CameraPoint(frame_count, 114, 190), CameraPoint(frame_count, 575, 828)),
#     car=geo_model.get_distance_from_camera_points(CameraPoint(frame_count, 1141, 712), CameraPoint(frame_count, 1446, 973))
# )
# depth_map_sanity_check = dict(
#     marked_points=depth_model.memo[frame_count][167, 296] - depth_model.memo[frame_count][967, 1142],
#     delineator=depth_model.memo[frame_count][190, 114] - depth_model.memo[frame_count][828, 575],
#     car=depth_model.memo[frame_count][712, 1141] - depth_model.memo[frame_count][973, 1446]
# )

class DepthModel:

    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.memo = {}

    def predict_depth(self, frame_id):
        if frame_id in self.memo: return self.memo[frame_id]

        if len(self.memo) > 10:
            depth_maps = [self.memo[frame] for frame in self.memo]
            # with open("object_detection_yolo/frames_detected/depth_map.npy", "wb") as fp:
            #     np.save(fp, sum(depth_maps)/len(depth_maps))
            return sum(depth_maps)/len(depth_maps)

        self.memo[frame_id] = load_depth_map_from_file(self.data_dir, max_depth=1, frame=frame_id)
        # predict depth here
        return self.memo[frame_id]


def get_intersection(line_a: Line, line_b: Line) -> Point:
    b = Point(*line_a.end.coords() - line_a.start.coords())
    d = Point(*line_b.end.coords() - line_b.start.coords())
    b_dot_d = b.x * d.y - b.y * d.x

    if b_dot_d == 0:
        # lines are parallel, no intersection
        return False
    
    c = Point(*line_b.start.coords() - line_a.start.coords())
    t = (c.x * d.y - c.y * d.x) / b_dot_d
    if t < 0 or t > 1:
        return False
    
    u = (c.x * b.y - c.y * b.x) / b_dot_d
    if u < 0 or u > 1:
        return False
    
    return Point(*line_a.start.coords() + t * b.coords())


def get_ground_truth_events(tracking_boxes, frame=None):
    # extract ground truth value for each tracking box
    ground_truth_events = []
    for object_id in tracking_boxes:
        center_points = np.array([(box.center_x, box.center_y) for box in tracking_boxes[object_id]])                
        if len(center_points) < 10 or len(center_points) > 750: 
            continue
        center_points_line = Line(Point(*center_points[0]), Point(*center_points[-1]))

        # extract ground truth value for each tracking box
        for box in tracking_boxes[object_id]:

            # check each of the for lines, spanned by the bounding box rectangle
            upper_line = Line(Point(box.x, box.y), Point(box.x+box.w, box.y))
            right_line = Line(Point(box.x+box.w, box.y), Point(box.x+box.w, box.y+box.h))
            lower_line = Line(Point(box.x, box.y+box.h), Point(box.x+box.w, box.y+box.h))
            left_line = Line(Point(box.x, box.y), Point(box.x, box.y+box.h))

            intersections = []
            for bounding_box_line in [upper_line, right_line, lower_line, left_line]:
                intersection = get_intersection(center_points_line, bounding_box_line)
                if intersection:
                    intersections.append(intersection)
            
            if len(intersections) == 2:
                # append ground truth only if line fully cuts bounding box
                intersect1, intersect2 = intersections
                ground_truth_events.append(
                    GroundTruthEvent(
                        (box.frame, int(intersect1.x), int(intersect1.y)), 
                        (box.frame, int(intersect2.x), int(intersect2.y)), 
                        6
                        )
                    )

                # cv2.rectangle(frame, (box.x, box.y), (box.x + box.w, box.y + box.h), (255, 0, 0), 2)
                # cv2.line(frame, (int(intersect1.x), int(intersect1.y)), (int(intersect2.x), int(intersect2.y)), (0,255,0), 8)
                # cv2.imwrite(f"object_detection_yolo/frames_detected/line_approach.jpg", frame)

    return ground_truth_events

def calculate_car_direction(car: Car) -> Direction:
    first_box = car.tracked_boxes[0]
    last_box = car.tracked_boxes[-1]

    if (first_box.y - last_box.y) < 0:
        return Direction.towards
    else:
        return Direction.away

def run(
    data_dir, fps=None, max_frames=None, custom_object_detection=False
):
    reload(logging)
    path_to_video = os.path.join(data_dir, "video.mp4")

    # Load Cars
    #cars = pd.read_csv(data_dir + "cars.csv")

    def avg_speed_for_time(timeStart, timeEnd):
        cars_to_avg = tracked_cars.loc[tracked_cars["start"].gt(timeStart) & tracked_cars["end"].le(timeEnd)]
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
        f'Run No.: {run_id}, Video: {data_dir}'
    )

    start = time.time()

    # Initialize Object Detection
    if custom_object_detection:
        weights = "object_detection_yolo/model_weights/yolov5/best.pt"
        od = ObjectDetectionCustom(weights=weights)
    else:
        od = ObjectDetection()

    input_video = cv2.VideoCapture(path_to_video)

    fps = give_me_fps(path_to_video) if fps is None else fps
    sliding_window = 60 * fps
    text_color = (255, 255, 255)
    # text_color = (0,0,0)

    # Initialize count
    frame_count = 0
    tracking_objects = {}
    frames = {}
    tracked_cars = {}
    track_id = 0
    avg_speed = "calculating"
    tracked_boxes = defaultdict(list)
    depth_model = DepthModel(data_dir)
    geo_model = GeometricModel(depth_model)
    is_calibrated = False

    # for shake_detection
    shake_detection = ShakeDetection()

    # meta_appr
    avg_frame_count = float(config.get("analyzer", "avg_frame_count"))
    speed_limit = int(config.get("analyzer", "speed_limit"))

    if custom_object_detection:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    while True:

        ############################
        # load frame, shake detection and object detection
        ############################
        ret, frame = input_video.read()
        frames[frame_count] = frame
        if frame_count == 0:
            # set normalization axes once at beginning
            c_u = int(frame.shape[1] / 2)
            c_v = int(frame.shape[0] / 2)
            geo_model.set_normalization_axes(c_u, c_v)

        if not ret:
            break

        if custom_object_detection:
            #cv2.imwrite(f"object_detection_yolo/frames_detected/frame_rgb.jpg", frame)
            frame = fgbg.apply(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            path_to_frame = f"object_detection_yolo/frames_detected/frame_{run_id}.jpg"
            cv2.imwrite(path_to_frame, frame)

        # for shake_detection
        if shake_detection.is_hard_move(frame):
            logging.info(
                f"Run No.: {run_id}, Video: {data_dir}, Hard Move Detected Frame: {frame_count}"
            )

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

        # Point current frame
        center_points_cur_frame = []

        for box in boxes:
            (x, y, w, h) = box.astype(int)
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)

            center_points_cur_frame.append(
                TrackingBox(cx, cy, x, y, w, h, frame_count)
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        ############################
        # assign tracking box IDs
        ############################
        for object_id, point_prev in tracking_objects.copy().items():
            object_exists = False
            for point_cur in center_points_cur_frame:
                distance = math.hypot(point_prev.x - point_cur.x, point_prev.y - point_cur.y)

                # Update IDs position
                if distance < 50:

                    tracking_objects[object_id] = point_cur
                    object_exists = True
                    if point_cur in center_points_cur_frame:
                        # Point should not match to multiple boxes
                        # TODO: Only take closest!
                        center_points_cur_frame.remove(point_cur)
                    break

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for point_cur in center_points_cur_frame:
            tracking_objects[track_id] = point_cur
            track_id += 1


        ############################
        # scaling factor estimation
        ############################
        if not is_calibrated:
            if len(tracked_boxes) > 150:
                # more than 10 cars were tracked
                ground_truth_events = get_ground_truth_events(tracked_boxes, frame)
                if len(ground_truth_events) > 50:
                    geo_model.scale_factor = offline_scaling_factor_estimation_from_least_squares(geo_model, ground_truth_events)
                    logging.info(f"Is calibrated: scale_factor: {geo_model.scale_factor}")
                    print(f"Is calibrated: scale_factor: {geo_model.scale_factor}", flush=True)
                    is_calibrated = True

            if frame_count % fps == 0:
                print(f"Have to calibrate: {len(tracked_boxes)}", flush=True)
            for object_id, point in tracking_objects.items():
                tracked_boxes[object_id].append(point)
        else:
            ############################
            # track cars
            ############################
            for object_id, point in tracking_objects.items():
                cv2.putText(frame, f"ID:{object_id}", (point.x + point.w + 5, point.y + point.h), 0, 1, (255,255,255), 2)
                if object_id in tracked_cars:
                    tracked_cars[object_id].tracked_boxes.append(point)
                    tracked_cars[object_id].frames_seen += 1
                    tracked_cars[object_id].frame_end += 1
                else:
                    

            ############################
            # speed estimation
            ############################
            if frame_count >= fps and frame_count % (60 * fps) == 0:
                # every 10 seconds
                car_count_towards = 0
                car_count_away = 0
                total_speed_towards = 0
                total_speed_away = 0
                total_speed_meta_appr_towards = 0
                total_speed_meta_appr_away = 0
                ids_to_drop = []

                for car_id, car in tracked_cars.items():

                    if (car.frame_end >= frame_count - sliding_window):
                        if car.frames_seen > 5:
                            car.direction = calculate_car_direction(car)
                            car_first_box = car.tracked_boxes[0]
                            car_last_box = car.tracked_boxes[-1] 
                            meters_moved = geo_model.get_distance_from_camera_points(
                                CameraPoint(car_first_box.frame, car_first_box.center_x, car_first_box.center_y), 
                                CameraPoint(car_last_box.frame, car_last_box.center_x, car_last_box.center_y)
                                )

                            if car.direction == Direction.towards:
                                car_count_towards += 1
                                total_speed_towards += (meters_moved) / (car.frames_seen / fps)
                                total_speed_meta_appr_towards += (avg_frame_count / int(car.frames_seen)) * speed_limit
                            else:
                                car_count_away += 1
                                total_speed_away += (meters_moved) / (car.frames_seen / fps)
                                total_speed_meta_appr_away += (avg_frame_count / int(car.frames_seen)) * speed_limit

                    else:
                        # car is too old, drop from tracked_cars
                        ids_to_drop.append(car_id)

                for car_id in ids_to_drop:
                    del tracked_cars[car_id]
                
                if car_count_towards > 0:
                    avg_speed = round((total_speed_towards / car_count_towards) * 3.6, 2)
                    print(f"Average speed towards: {avg_speed} km/h")
                    print(f"Average META speed towards: {(total_speed_meta_appr_towards / car_count_towards)} km/h")
                    logging.info(json.dumps(dict(frameId=frame_count, avgSpeedTowards=avg_speed)))

                if car_count_away > 0:
                    avg_speed = round((total_speed_away / car_count_away) * 3.6, 2)
                    print(f"Average speed away: {avg_speed} km/h")
                    print(f"Average META speed away: {(total_speed_meta_appr_away / car_count_away)} km/h")
                    logging.info(json.dumps(dict(frameId=frame_count, avgSpeedAway=avg_speed)))



        ############################
        # output text on video stream
        ############################
        cv2.putText(
            frame, f"Timestamp: {(frame_count / fps):.2f} s", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2,
        )
        cv2.putText(
            frame, f"FPS: {fps}", (7, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2
        )
        cv2.imwrite(f"object_detection_yolo/frames_detected/frame_after_detection.jpg", frame)

        if frame_count % 500 == 0:
            print(
                f"Frame no. {frame_count} time since start: {(time.time()-start):.2f}s"
            )
        frame_count += 1
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

    session_path_local = sys.argv[1] if len(sys.argv) > 1 else session_path
    log_name = run(
        session_path_local,
        fps,
        max_frames=max_frames,
        custom_object_detection=custom_object_detection,
    )


    ### Evaluation
    plot_absolute_error([log_name], "logs/")


if __name__ == "__main__":
    main()
