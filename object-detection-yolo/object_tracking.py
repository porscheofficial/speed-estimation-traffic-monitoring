import cv2
import numpy as np
from object_detection import ObjectDetection
#from moviepy.editor import *
from pre_pro_fps_ppm import get_fps_and_ppm
import math
import os
from get_fps import give_me_fps
import pandas as pd
import logging
import json
from sklearn.metrics.pairwise import euclidean_distances

logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

cars_path = "/Users/saver/Documents/master/semester-1/AIP/brno_complete/2016-ITS-BrnoCompSpeed/dataset/session0_right/"

path_to_dataset = "datasets/yt_video1.mp4"
path_to_dataset = "datasets/test.m4v"
path_to_dataset = cars_path + "video.mp4"
#path_to_dataset = "../datasets/test.m4v"
# path_to_dataset = "los_angeles.mp4"

sliding_window = 15

#text_color = (255,255,255)
text_color = (0,0,0)

cars = pd.read_csv(cars_path + "cars.csv")

def avg_speed_for_time(timeStart, timeEnd):
    cars_to_avg = cars.loc[cars['start'].gt(timeStart) & cars['end'].le(timeEnd)]
    return cars_to_avg['speed'].mean()

class Car:
    def __init__(self, pixels_moved, frames_seen, frame_start, frame_end) -> None:
        self.pixels_moved = pixels_moved
        self.frames_seen = frames_seen
        self.frame_start = frame_start
        self.frame_end = frame_end


class Point:
    def __init__(self, x, y, ppm) -> None:
        self.x = x
        self.y = y
        self.ppm = ppm

def run():
    # Initialize Object Detection
    od = ObjectDetection()

    cap = cv2.VideoCapture(path_to_dataset)

    fps, ppm = get_fps_and_ppm(path_to_dataset)
    fps = give_me_fps(path_to_dataset)
    ppm = 55

    # fps = get_fps_from_video("../datasets/yt_video1.mp4")

    # Initialize count
    frame_count = 0

    tracking_objects = {}
    tracking_objects_prev = {}
    cars = {}
    track_id = 0
    avg_speed = "calculating"

    while True:
        ret, frame = cap.read()
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
            center_points_cur_frame.append(Point(cx, cy, w/1.9))
            # print("FRAME N°", count, " ", x, y, w, h)

            # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                    frame,
                    f"Breite: {w:.2f}",
                    (int(x + w), int(y + h)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    (255, 255, 255),
                    1,
                )


        # Only at the beginning we compare previous and current frame
        if frame_count <= 2:
            for point in center_points_cur_frame:
                for point2 in center_points_prev_frame:
                    distance = math.hypot(point2.x - point.x, point2.y - point.y)

                    if distance < 20:
                        tracking_objects[track_id] = point
                        track_id += 1
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
                        continue

                # Remove IDs lost
                if not object_exists:
                    tracking_objects.pop(object_id)

            # Add new IDs found
            for point in center_points_cur_frame:
                tracking_objects[track_id] = point
                track_id += 1

        for object_id, point in tracking_objects.items():
            if object_id in tracking_objects_prev:
                x_movement = tracking_objects[object_id].x - tracking_objects_prev[object_id].x
                y_movement = tracking_objects[object_id].y - tracking_objects_prev[object_id].y
                avg_ppm = (tracking_objects[object_id].ppm + tracking_objects_prev[object_id].ppm)/2

                total_movement = math.sqrt(math.pow(x_movement, 2) + math.pow(y_movement, 2))

                cv2.putText(frame, str(total_movement), (point.x, point.y - 30), 0, 1, (0, 0, 255), 2)
                logging.info(json.dumps(dict(fps=fps, carId=object_id, ppm=avg_ppm, metersMoved=total_movement/avg_ppm,
                pixelsMoved=total_movement)))

                if object_id in cars:
                    cars[object_id].pixels_moved = total_movement/avg_ppm
                    cars[object_id].frames_seen = 1
                    cars[object_id].frame_end += 1                    
                else:
                    cars[object_id] = Car(total_movement, 1, frame_count, frame_count)

            else:
                cv2.putText(frame, str(-1), (point.x, point.y - 30), 0, 1, (0, 0, 255), 2)

            cv2.circle(frame, (point.x, point.y), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (point.x, point.y - 7), 0, 1, (0, 0, 255), 2)


        # Make a copy of the points
        center_points_prev_frame = center_points_cur_frame.copy()
        tracking_objects_prev = tracking_objects.copy()


        cv2.putText(
            frame,
            f"Timestamp: {(frame_count / fps):.2f} s",
            (7, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2,
        )

        key = cv2.waitKey(1)
        if key == 27:
            break
        
        
        # if count >= 15 * fps/15 and count % (5 * fps/10) == 0:

        #     total_speed = 0
        #     car_count = 0

        #     for car in cars.values():

        #         if car.frame_end >= count - sliding_window:
        #             car_count += 1
        #             total_speed += (car.pixels_moved / ppm) / (car.frames_seen / fps)

        #     print(f"Average speed: {(total_speed / car_count):.2f} m/s")
        #     print(f"Average speed: {(total_speed / car_count) * 3.6:.2f} km/h")
        #     avg_speed = f"{(total_speed / car_count) * 3.6:.2f}"



        total_speed = 0
        car_count = 0

        if len(cars) >0:
            for car in cars.values():

                if car.pixels_moved > 0.05:
                    car_count += 1
                    total_speed += (car.pixels_moved) / (car.frames_seen / fps)
            if car_count > 0:
                print(f"Average speed: {(total_speed / car_count):.2f} m/s")
                print(f"Average speed: {(total_speed / car_count) * 3.6:.2f} km/h")
                avg_speed = f"{(total_speed / car_count) * 3.6:.2f}"

        cv2.putText(
            frame,
            f"Average speed: {avg_speed} km/h",
            (7, 90),    
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2
        )
        cv2.putText(
            frame,
            f"Ground truth: {avg_speed_for_time(0,10):.2f} km/h",
            (7, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2
        )
        cv2.putText(
            frame,
            f"FPS: {fps}",
            (7, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2
        )

        cv2.imshow("Frame", frame)
        cv2.imwrite("frames_detected/frame%d.jpg" % frame_count, frame)

    #render_detected_frames_to_video(count, fps, 'detected.mp4', 'frames_detected/frame%d.jpg')
    #count_birds_eye = transform_to_birds_eye('detected.mp4')
    #render_detected_frames_to_video(count_birds_eye, fps, 'birdseye.mp4', 'birds_eye_frames/birdframe%d.jpg')

    cap.release()
    cv2.destroyAllWindows()


def render_detected_frames_to_video(count, fps, out_video_name, path_to_frames):
    img_array = []
    for c in range(0, count):
        c += 1
        img = cv2.imread(path_to_frames % c)

        if img is None:
            continue

        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc(*'MP4V'), fps,
                          size)  # fps have to get set automatically from orignal video
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def get_fps_from_video(path_to_video):
    # loading video dsa gfg intro video
    # clip = VideoFileClip(path_to_video).subclip(0, 10)
    clip = VideoFileClip(path_to_video)

    # getting frame rate of the clip
    return clip.fps


if __name__ == '__main__':
    run()
