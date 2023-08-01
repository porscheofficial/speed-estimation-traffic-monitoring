"""Saves videos from USA transportation web.

This file can be used to download videos from the USA transportation web.
"""

from csv import reader

import cv2 as cv
import ffmpeg


def download_video(input_url: str, t: int = 180):
    """Download video.

    Download a video from the given url.

    @param input_url:
        Url to the video that should be downloaded.
    @param t:
        Duration of the video to be saved.
    @return:
        None.
    """
    output_path = f"{input_url.split('/')[-2]}.mp4"
    stream = ffmpeg.input(input_url)
    stream = ffmpeg.output(stream, filename=output_path, t=t, loglevel="quiet")
    ffmpeg.run(stream, overwrite_output=True)
    return  # f"File saved here in {output_path}"


# skip the first line as it is the head line
with open("urls1_.csv") as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)

    if header != None:
        for row in csv_reader:
            print(row)
            url = row[2]
            print(url)
            cap = cv.VideoCapture(url)
            if cap is None or not cap.isOpened():
                print("unable to open the source:", url)
            else:
                download_video(url)
