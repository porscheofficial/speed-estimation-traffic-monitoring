import cv2
import subprocess as sp
import numpy
import argparse
import time
import math


def get_video_stream_from_url(url):
    # adjust these when run on windows:
    FFMPEG_BIN = "ffmpeg"
    FFPROBE_BIN = "ffprobe"

    # get the dimensions of the stream for reading the real stream later
    meta_pipe = sp.Popen([FFPROBE_BIN, "-v", "error",
                          "-select_streams", "v:0",
                          "-show_entries", "stream=width,height",  # disable audio
                          "-of", "csv=p=0",
                          url],
                         stdin=sp.PIPE, stdout=sp.PIPE)
    frame_size = meta_pipe.stdout.read().decode("utf-8").split("\n")[0].split(",")

    # get the actual stream through ffmpeg and return it
    stream_pipe = sp.Popen([FFMPEG_BIN, "-i", url,
                            "-loglevel", "quiet",  # no text output
                            "-an",  # disable audio
                            "-f", "image2pipe",
                            "-pix_fmt", "bgr24",
                            "-vcodec", "rawvideo", "-"],
                           stdin=sp.PIPE, stdout=sp.PIPE)
    return stream_pipe, int(frame_size[0]), int(frame_size[1])


# helper function to mask handling the input
def retrieve_next_frame_from_stream(pipe, w, h):
    raw_image = pipe.stdout.read(w * h * 3)
    return numpy.frombuffer(raw_image, numpy.uint8).reshape((h, w, 3))


# function to estimate fps for a stream from 1000 frames
def count_fps_from_stream(pipe, w, h):
    n_frames = 1000
    start_time = time.time()
    for _ in range(n_frames):
        retrieve_next_frame_from_stream(pipe, w, h)
    end_time = time.time()
    delta_t = end_time - start_time
    return n_frames / delta_t


def downsample_fps(max, fps):
    threshold = max / (fps - max)
    new_fps = fps - int(fps / math.floor(threshold+1))
    return int(new_fps), threshold

# main method is only for viewing and testing the stream
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show an m3u8 stream over http')
    parser.add_argument(
        'stream_link',
        type=str,
        help='Enter the link to an m3u8 stream over http',
    )

    # constant for setting max fps for down-sampling
    MAX_FPS = 30

    stream, width, height = get_video_stream_from_url(parser.parse_args().stream_link)
    fps = count_fps_from_stream(stream, width, height)

    if fps > MAX_FPS:
        print("fps over ", MAX_FPS, ": ", fps)
        real_fps, thresh = downsample_fps(MAX_FPS, fps)
        i = 0
        while True:
            if thresh <= i:
                i = 0
            else:
                cv2.imshow("Stream", retrieve_next_frame_from_stream(stream, width, height))
                i += 1
            if cv2.waitKey(int(1000 / real_fps)) == ord('q'):
                break
    else:

        print("fps under ", MAX_FPS, ": ", fps)
        while True:
            cv2.imshow("Stream", retrieve_next_frame_from_stream(stream, width, height))
            if cv2.waitKey(int(1000 / fps)) == ord('q'):
                break

    cv2.destroyAllWindows()
