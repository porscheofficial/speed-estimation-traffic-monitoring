import cv2


def clamp(n, smallest, largest): return max(smallest, min(n, largest)) 

class Car:
    def __init__(self, meters_moved, frames_seen, frame_start, frame_end) -> None:
        self.meters_moved = meters_moved
        self.frames_seen = frames_seen
        self.frame_start = frame_start
        self.frame_end = frame_end

class Point:
    def __init__(self, center_x, center_y, meters_moved, x, y, w, h, frame, ppm) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.meters_moved = meters_moved
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame = frame
        self.ppm = ppm

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
