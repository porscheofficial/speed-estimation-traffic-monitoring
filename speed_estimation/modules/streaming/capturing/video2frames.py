"""Helper file.

NOTE: THIS FILE WAS USED AS A HELPER FILE DURING DEVELOPMENT OF THE PIPELINE.
IT IS NOT INTEGRATED WITHIN THE PIPELINE.
"""

import glob
import os

import cv2

videoPath = "/Users/p393919/Documents/os/HamptonRoads/*.mp4"
videos = glob.glob(videoPath)

j = 0

for video in sorted(videos):
    fileName = os.path.basename(video)
    fileName = os.path.splitext(fileName)[0]

    parentDir = "/Users/p393919/Documents/os/frames"

    newDir = os.path.join(parentDir, fileName)
    os.mkdir(newDir)

    inputpath = os.path.abspath(video)

    outpath = os.path.abspath(newDir)

    # Opens the Video file
    cap = cv2.VideoCapture(inputpath)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(
            os.path.join(outpath, fileName + "_" + str(i).zfill(4) + ".png"), frame
        )
        i += 1

    cap.release()
    cv2.destroyAllWindows()

    j = j + 1
