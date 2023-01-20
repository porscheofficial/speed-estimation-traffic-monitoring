import cv2
import subprocess as sp
import numpy

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"
VIDEO_URL = "https://49-d2.divas.cloud/CHAN-8293/CHAN-8293_1.stream/playlist.m3u8?217.232.115.59&vdswztokenhash=M27XmYvnp9yZ4EQO-Gq1GQbSCYhwCMj6LlGrX_vBiuc="

meta_pipe = sp.Popen([FFPROBE_BIN, "-v", "error",
                      "-select_streams", "v:0",
                      "-show_entries", "stream=width,height",  # disable audio
                      "-of", "csv=p=0",
                      VIDEO_URL],
                     stdin=sp.PIPE, stdout=sp.PIPE)
frame_size = meta_pipe.stdout.read().decode("utf-8").split("\n")[0].split(",")
width = int(frame_size[0])
height = int(frame_size[1])

stream_pipe = sp.Popen([FFMPEG_BIN, "-i", VIDEO_URL,
                        "-loglevel", "quiet",  # no text output
                        "-an",  # disable audio
                        "-f", "image2pipe",
                        "-pix_fmt", "bgr24",
                        "-vcodec", "rawvideo", "-"],
                       stdin=sp.PIPE, stdout=sp.PIPE)
while True:
    raw_image = stream_pipe.stdout.read(width * height * 3)
    image = numpy.frombuffer(raw_image, numpy.uint8).reshape((height, width, 3))
    cv2.imshow("Stream", image)
    if cv2.waitKey(30) == ord('q'):
        break

cv2.destroyAllWindows()

# cap = cv2.VideoCapture('https://www.bogotobogo.com/VideoStreaming/VLC/Images/SimpleHttpStreaming/planet_small.mp4')

# if not cap.isOpened():
#    print("Cannot open stream")
#    exit()

# while True:
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     cv2.imshow('live cam', gray)
#     if cv2.waitKey(25) == ord('q'):
#         break

# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
