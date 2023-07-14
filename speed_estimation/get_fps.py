"""
This module holds functions to derive the frames per second (fps) from the video or stream that
should be analyzed.
This module will only be used if the FPS are not defined in speed_estimation/config.ini
"""
import time

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip


def get_fps_from_video(path_to_video: str) -> int:
    """Get fps of video/stream.

    The function uses the library moviepy to derive the fps.

    @param path_to_video:
        The path to the video whose fps are to be found out.
    @return:
        The fps are returned.
    """
    # loading video dsa gfg intro video
    clip = VideoFileClip(path_to_video)

    # getting frame rate of the clip
    return clip.fps


def read_fps_cv2(path_to_video: str) -> int:
    """Get fps of video/stream.

    The function uses opencv to derive the fps.

    @param path_to_video:
        The path to the video whose fps are to be found out.
    @return:
        The fps are returned.
    """
    video = cv2.VideoCapture(path_to_video)
    (major_ver, _, _) = cv2.__version__.split(".")

    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)

    video.release()
    return fps


def read_fps_manually(path_to_video: str) -> int:
    """Get fps of video/stream.

    The function calculates the fps manually.
    Therefore, the fps will be number of frame processed in given time frame
    Since there will be most of the time a error of 0.001 second we will be subtracting it to get
    more accurate result.

    @param path_to_video:
        The path to the video whose fps are to be found out.
    @return:
        The fps are returned.
    """
    cap = cv2.VideoCapture(path_to_video)

    # used to record the time when we processed last frame
    prev_frame_time = 0.0

    # used to record the time at which we processed current frame
    new_frame_time = 0.0

    fps_list = []
    counter = 0
    while cap.isOpened():
        counter += 1

        if counter == 20:
            break

        # Capture frame-by-frame
        ret, _ = cap.read()

        # if video finished or no Video Input
        if not ret:
            break

        # time when we finish processing for this frame
        new_frame_time = time.time()

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)
        fps_list.append(fps)

    cap.release()

    avg_fps = int(sum(fps_list) / len(fps_list))

    return avg_fps


def get_fps(path_to_dataset):
    """Get fps of video/stream.

    The function uses two different approaches to derive the fps.
    The most plausible solution is returned.

    @param path_to_video:
        The path to the video whose fps are to be found out.
    @return:
        The fps are returned.
    """
    fps_a = read_fps_cv2(path_to_dataset)
    fps_b = read_fps_manually(path_to_dataset)
    fps = fps_a
    print("a:" + str(fps_a))
    print("b:" + str(fps_b))

    if fps > 80:
        fps = fps_b
        if fps > 80:
            fps = 25  # average of most videos

    print("final:" + str(fps))
    return int(fps)
