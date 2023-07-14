"""
This module helps you to stream video footage and directly analyze it with the speed estimation
pipeline.
The main function that should be called therefore is get_video_stream_from_url. Since the pipline
per default operates on videos you have to replace the loading of the video with this method.
"""
import argparse
import math
import subprocess as sp
import time

import cv2
import numpy

from numpy.typing import NDArray


class StreamHandler:
    """
    This class can be used when the speed estimation should be applied to a video stream instead of
    a video.
    """

    def __init__(self, url: str) -> None:
        """Create an instance of the StreamHandler.

        @param url:
            URL where the stream can be reached.
        """
        self.url = url
        # adjust these when run on windows:
        self.FFMPEG_BIN = "ffmpeg"
        self.FFPROBE_BIN = "ffprobe"

    def run(self, visually: bool = False) -> NDArray:
        """Get the next frame in the stream

        @param visually:
            For demo purpose set this to True. It will prompt a window with the stream. Press q to
            close the window.
        @return:
            The next frame in the stream.
        """
        stream, width, height = stream_handler.__get_video_stream_from_url()
        fps = self.__count_fps_from_stream(stream, width, height)

        next_frame = self.__retrieve_next_frame_from_stream(stream, width, height)

        if visually:
            while True:
                next_frame_display = self.__retrieve_next_frame_from_stream(
                    stream, width, height
                )

                if fps > MAX_FPS:
                    real_fps, thresh = self.__downsample_fps(MAX_FPS, fps)
                    i = 0

                    if thresh <= i:
                        i = 0
                    else:
                        cv2.imshow("Stream", next_frame_display)
                        i += 1
                else:
                    cv2.imshow("Stream", next_frame_display)

                if cv2.waitKey(int(1000 / fps)) == ord("q"):
                    break

            cv2.destroyAllWindows()

        return next_frame

    def __get_video_stream_from_url(self) -> tuple[sp.Popen, int, int]:
        """Retrieve a video stream from the given url.

        @param url:
            The URL where the stream can be reached.

        @return:
            The method returns a Popen object holding the stream, and the expected width and height
            of the frames extracted from the stream.
        """

        # get the dimensions of the stream for reading the real stream later
        meta_pipe = sp.Popen(
            [
                self.FFPROBE_BIN,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",  # disable audio
                "-of",
                "csv=p=0",
                self.url,
            ],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
        )

        frame_size = meta_pipe.stdout.read().decode("utf-8").split("\n")[0].split(",")

        # get the actual stream through ffmpeg and return it
        stream_pipe = sp.Popen(
            [
                self.FFMPEG_BIN,
                "-i",
                self.url,
                "-loglevel",
                "quiet",  # no text output
                "-an",  # disable audio
                "-f",
                "image2pipe",
                "-pix_fmt",
                "bgr24",
                "-vcodec",
                "rawvideo",
                "-",
            ],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
        )

        return stream_pipe, int(frame_size[0]), int(frame_size[1])

    def __retrieve_next_frame_from_stream(
        self, stream: sp.Popen, w: int, h: int
    ) -> NDArray:
        """Derive the next stream from the stream.

        @param stream:
            The Popen object holding the stream.
        @param w:
            The width of the stream frames
        @param h:
            The height of the stream frames.
        @return:
            The next frame.
        """
        raw_image = stream.stdout.read(w * h * 3)
        return numpy.frombuffer(raw_image, numpy.uint8).reshape((h, w, 3))

    # function to estimate fps for a stream from 1000 frames
    def __count_fps_from_stream(self, stream, w, h):
        n_frames = 1000
        start_time = time.time()
        for _ in range(n_frames):
            self.__retrieve_next_frame_from_stream(stream, w, h)
        end_time = time.time()
        delta_t = end_time - start_time
        return n_frames / delta_t

    def __downsample_fps(self, max, fps):
        threshold = max / (fps - max)
        new_fps = fps - int(fps / math.floor(threshold + 1))

        return (int(new_fps), threshold) if int(new_fps) > 0 else (max, threshold)


# main method is only for viewing and testing the stream
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show an m3u8 stream over http")
    parser.add_argument(
        "--stream_link",
        type=str,
        help="Enter the link to an m3u8 stream over http",
    )

    # constant for setting max fps for down-sampling
    MAX_FPS = 30

    stream_handler = StreamHandler(parser.parse_args().stream_link)
    stream_handler.run(visually=True)
