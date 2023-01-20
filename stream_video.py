import cv2
import subprocess as sp
import numpy
import argparse
import time


def get_video_stream_from_url(url):
    FFMPEG_BIN = "ffmpeg"
    FFPROBE_BIN = "ffprobe"

    meta_pipe = sp.Popen([FFPROBE_BIN, "-v", "error",
                          "-select_streams", "v:0",
                          "-show_entries", "stream=width,height",  # disable audio
                          "-of", "csv=p=0",
                          url],
                         stdin=sp.PIPE, stdout=sp.PIPE)
    frame_size = meta_pipe.stdout.read().decode("utf-8").split("\n")[0].split(",")

    stream_pipe = sp.Popen([FFMPEG_BIN, "-i", url,
                            "-loglevel", "quiet",  # no text output
                            "-an",  # disable audio
                            "-f", "image2pipe",
                            "-pix_fmt", "bgr24",
                            "-vcodec", "rawvideo", "-"],
                           stdin=sp.PIPE, stdout=sp.PIPE)
    return stream_pipe, int(frame_size[0]), int(frame_size[1])


def retrieve_next_frame_from_stream(pipe, w, h):
    raw_image = pipe.stdout.read(w * h * 3)
    return numpy.frombuffer(raw_image, numpy.uint8).reshape((h, w, 3))


def count_fps_from_stream(pipe, w, h):
    n_frames = 1000
    start_time = time.time()
    for _ in range(n_frames):
        retrieve_next_frame_from_stream(pipe, w, h)
    end_time = time.time()
    delta_t = end_time - start_time
    return n_frames / delta_t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show an m3u8 stream over http')
    parser.add_argument(
        'stream_link',
        type=str,
        help='Enter the link to an m3u8 stream over http',
    )

    stream, width, height = get_video_stream_from_url(parser.parse_args().stream_link)
    fps = count_fps_from_stream(stream, width, height)

    while True:
        cv2.imshow("Stream", retrieve_next_frame_from_stream(stream, width, height))
        if cv2.waitKey(int(1000 / fps)) == ord('q'):
            break

    cv2.destroyAllWindows()
