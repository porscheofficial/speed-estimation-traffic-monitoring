import cv2
from tracker import *

#create tracker object
tracker = EuclideanDistTracker()


# would be great if it would not move
cap = cv2.VideoCapture("../datasets/output_ori.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=180) #super high threshold since less false positiv

while True:
    ret, frame = cap.read()
    #maybe define region of interest (hard automatically)
    #1 obj detection
    mask = object_detector.apply(frame)
    #remove some noise
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)



    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        # calc area and remove the smaller elements
        area = cv2.contourArea(cnt)
        if area > 1000 and area < 20000: #pixels of too small /big stuff, should be automaticall i guess
            #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2) #shows real detected cotours
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y + h), (0, 255, 0), 3) #shows rectangles of contours

            detections.append([x, y, w, h])

    #2. object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2) #to attach id to a little above the object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    mask = cv2.blur(mask, (20, 20), 0)
    cv2.imshow("mask", mask)

    #cv2.imshow("Frame", frame)

    key = cv2.waitKey(30) #waits 30 ms between each frame
    if key == 27: #press esc to kill
        break

cap.release()
cv2.destroyAllWindows()