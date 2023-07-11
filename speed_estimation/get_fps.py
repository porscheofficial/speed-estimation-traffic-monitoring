import time

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip


def get_fps_from_video(path_to_video):
    # loading video dsa gfg intro video
    # clip = VideoFileClip(path_to_video).subclip(0, 10)
    clip = VideoFileClip(path_to_video)

    # getting frame rate of the clip
    return clip.fps


def give_me_fps(path_to_dataset):
    fps_a = read_fps_cv2(path_to_dataset)
    fps_b = read_fps_strange(path_to_dataset)
    fps = fps_a
    print("a:" + str(fps_a))
    print("b:" + str(fps_b))

    if fps > 80:
        fps = fps_b
        if fps > 80:
            fps = 25  # average of most videos

    print("final:" + str(fps))
    return int(fps)


# In[2]:


def read_fps_cv2(path_to_dataset):
    video = cv2.VideoCapture(path_to_dataset)
    (major_ver, _, _) = cv2.__version__.split(".")

    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        # print (fps)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        # print (fps)

    video.release()
    return fps


# In[10]:


def read_fps_strange(path_to_dataset):
    cap = cv2.VideoCapture(path_to_dataset)

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

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

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)
        fps_list.append(fps)

    cap.release()

    avg_fps = sum(fps_list) / len(fps_list)
    return avg_fps
