from datetime import datetime
from importlib import reload
from multiprocessing import Pool
import cv2
import os

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

from moviepy.video.io.VideoFileClip import VideoFileClip

from object_detection import ObjectDetection
from pre_pro_fps_ppm import get_fps_and_ppm
import math
from get_fps import give_me_fps
import pandas as pd
import logging
import json
from paths import cars_path, path_to_dataset
import copy
import numpy as np
import imutils
import time
import uuid



# white
text_color = (255,255,255)
# black
#text_color = (0,0,0)


def clamp(n, smallest, largest): return max(smallest, min(n, largest)) 


class Car:
    def __init__(self, meters_moved, frames_seen, frame_start, frame_end) -> None:
        self.meters_moved = meters_moved
        self.frames_seen = frames_seen
        self.frame_start = frame_start
        self.frame_end = frame_end


class Point:
    def __init__(self, center_x, center_y, meters_moved, x, y, w, h, frame) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.meters_moved = meters_moved
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame = frame

# TODO: Move to depth map module
def extract_frame(input: str, output_folder: str, output_file: str, frame_idx: int = 0) -> str:
    input_video = cv2.VideoCapture(input)
    frame_count = 0
    while True:
        ret, frame = input_video.read()
        #frame = frame[890-352:890]
        #frame = image_resize(frame, width=1900)
        #frame = cv2.resize(frame, (1900, 1056))
        frame = imutils.resize(frame, height=352)
        frame = cv2.copyMakeBorder(frame, left=295, right=296, top=0, bottom=0, borderType=cv2.BORDER_CONSTANT)
        if frame_idx == frame_count:
            path = os.path.join(output_folder, output_file % frame_idx)
            cv2.imwrite(path, frame)
            return output_file % frame_idx
        frame_count += 1

# TODO: Move to depth map module
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

def run(video_folder=None, max_depth=None):
    reload(logging)
    current_folder = cars_path if video_folder is None else video_folder
    path_to_video = path_to_dataset if video_folder is None else os.path.join(video_folder, 'video.mp4')

    # Load Cars
    cars = pd.read_csv(current_folder + "cars.csv")

    def avg_speed_for_time(timeStart, timeEnd):
        cars_to_avg = cars.loc[cars['start'].gt(timeStart) & cars['end'].le(timeEnd)]
        return cars_to_avg['speed'].mean()

    run_id = uuid.uuid4().hex[:10]
    print(f"Run No.: {run_id}")

    # Initialize logging
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f'logs/{now_str}_run_{run_id}.log', level=logging.DEBUG)
    logging.info(f'Run No.: {run_id}, Video: {current_folder}, Max Depth: {"None" if max_depth is None else max_depth}')

    # Load depth map
    our_meters = load_depth_map(current_folder, max_depth=max_depth)

    # Initialize Object Detection
    start = time.time()
    od = ObjectDetection()

    input_video = cv2.VideoCapture(path_to_video)

    #fps, ppm = get_fps_and_ppm(path_to_dataset)
    fps = give_me_fps(path_to_video)

    fps = 50
    sliding_window = 15
    sliding_window *= fps

    # Initialize count
    frame_count = 0

    tracking_objects = {}
    tracking_objects_prev = {}
    cars = {}
    track_id = 0
    avg_speed = "calculating"

    while True:
        ret, frame = input_video.read()        
        frame_count += 1
        if not ret:
            break
        #frame = frame[890-352:890]
        #frame = image_resize(frame, width=1900)
        #frame = cv2.resize(frame, (1900, 1056))
        frame = imutils.resize(frame, height=352)
        frame = cv2.copyMakeBorder(frame, left=295, right=296, top=0, bottom=0, borderType=cv2.BORDER_CONSTANT)
        # cv2.imwrite("object-detection-yolo/frames_detected/frame%d_new_scaled.jpg" % frame_count, frame)
        # Point current frame
        center_points_cur_frame = []
        center_points_prev_frame = []

        # Detect objects on frame
        (class_ids, scores, boxes) = od.detect(frame)
        for box in boxes:
            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            if cy < our_meters.shape[0] and cx < our_meters.shape[1]:
                meters = our_meters[cy][cx]
            else:
                meters = 0
            center_points_cur_frame.append(Point(cx, cy, meters, x, y, w, h, frame_count))

            # print("FRAME N°", count, " ", x, y, w, h)

            # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            # cv2.putText(
            #         frame,
            #         f"Breite: {w:.2f}",
            #         (int(x + w), int(y + h)),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.95,
            #         (255, 255, 255),
            #         1,
            #     )

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
            meters_moved = None
            if object_id in tracking_objects_prev:
                x_movement = tracking_objects[object_id].x - tracking_objects_prev[object_id].x
                y_movement = tracking_objects[object_id].y - tracking_objects_prev[object_id].y
                total_movement = math.sqrt(math.pow(x_movement, 2) + math.pow(y_movement, 2))

                meters_moved = abs(tracking_objects[object_id].meters_moved - tracking_objects_prev[object_id].meters_moved)
                #cv2.putText(frame, str(total_movement), (point.x, point.y - 30), 0, 1, (0, 0, 255), 2)
                #logging.info(json.dumps(dict(fps=fps, frameId=frame_count, carId=object_id, metersMoved=str(meters_moved))))

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
                # logging.info(json.dumps(dict(frameId=frame_count, carId=car_id, 
                #                             metersMoved=str(car.meters_moved), 
                #                             framesSeen=car.frames_seen,
                #                             frameStart=car.frame_start,
                #                             frameEnd=car.frame_end)))

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
                # logging.info(json.dumps(dict(frameId=frame_count, carId=car_id, 
                #                             metersMoved=str(car.meters_moved), 
                #                             framesSeen=car.frames_seen,
                #                             frameStart=car.frame_start,
                #                             frameEnd=car.frame_end)))

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

    #render_detected_frames_to_video(frame_count, fps, 'detected.avi', 'object-detection-yolo/frames_detected/frame%d.jpg')
    #count_birds_eye = transform_to_birds_eye('detected.mp4')
    #render_detected_frames_to_video(count_birds_eye, fps, 'birdseye.mp4', 'birds_eye_frames/birdframe%d.jpg')

    input_video.release()
    cv2.destroyAllWindows()
    logging.shutdown()


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

    out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps,
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
    # jobs_to_run = [
    #     ("/scratch2/video_samples/session6_left/", 80),
    #     ("/scratch2/video_samples/session6_left/", 100),
    #     ("/scratch2/video_samples/session6_left/", 120),
    # ]

    # with Pool(processes=8) as pool:
    #     pool.starmap(run, jobs_to_run)
    video = "/scratch2/video_samples/session4_right/"
    # video = "/scratch2/video_samples/session1_center/"
    # video = "/scratch2/video_samples/session6_left/"
    
    run(video, 80)
    run(video, 100)
    run(video, 120)
