import cv2

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 

video_cap = cv2.VideoCapture("datasets/video_session1_center.mp4")
image_prefix = "s1c_"

counter = 0
frame_step = 500

while True:
    success, frame = video_cap.read()

    if counter % frame_step == 0:
        # applying on each frame
        fgmask = fgbg.apply(frame)

        #cv2.imshow("frame", fgmask)
        if fgmask is not None:
            cv2.imwrite("background/" + image_prefix + str(counter) + ".png", fgmask)
    
    counter += 1