import cv2
import numpy as np
from object_detection import ObjectDetection
from autobird import transform_to_birds_eye
import math
import os

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
from moviepy.editor import *


def run():
    # Initialize Object Detection
    od = ObjectDetection()

    cap = cv2.VideoCapture("ori.avi")

    fps = get_fps_from_video("ori.avi")

    # Initialize count
    count = 0
    center_points_prev_frame = []

    tracking_objects = {}
    track_id = 0

    while True:
        ret, frame = cap.read()
        count += 1
        if not ret:
            break

        # Point current frame
        center_points_cur_frame = []

        # Detect objects on frame
        (class_ids, scores, boxes) = od.detect(frame)
        for box in boxes:
            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))
            # print("FRAME N°", count, " ", x, y, w, h)

            # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Only at the beginning we compare previous and current frame
        if count <= 2:
            for pt in center_points_cur_frame:
                for pt2 in center_points_prev_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    if distance < 20:
                        tracking_objects[track_id] = pt
                        track_id += 1
        else:

            tracking_objects_copy = tracking_objects.copy()
            center_points_cur_frame_copy = center_points_cur_frame.copy()

            for object_id, pt2 in tracking_objects_copy.items():
                object_exists = False
                for pt in center_points_cur_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    # Update IDs position
                    if distance < 20:
                        tracking_objects[object_id] = pt
                        object_exists = True
                        if pt in center_points_cur_frame:
                            center_points_cur_frame.remove(pt)
                        continue

                # Remove IDs lost
                if not object_exists:
                    tracking_objects.pop(object_id)

            # Add new IDs found
            for pt in center_points_cur_frame:
                tracking_objects[track_id] = pt
                track_id += 1

        for object_id, pt in tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

        print("Tracking objects")
        print(tracking_objects)

        print("CUR FRAME LEFT PTS")
        print(center_points_cur_frame)

        cv2.imshow("Frame", frame)
        cv2.imwrite("frames_detected/frame%d.jpg" % count, frame)

        # Make a copy of the points
        center_points_prev_frame = center_points_cur_frame.copy()

        key = cv2.waitKey(1)
        if key == 27:
            break

    render_detected_frames_to_video(count, fps, 'detected.mp4', 'frames_detected/frame%d.jpg')
    count_birds_eye = transform_to_birds_eye('detected.mp4')
    render_detected_frames_to_video(count_birds_eye, fps, 'birdseye.mp4', 'birds_eye_frames/birdframe%d.jpg')

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
