# %%
import cv2
import dlib
import time


def get_fps_and_ppm(path_to_dataset):
    # haar cascade for vehicle tracking
    carCascade = cv2.CascadeClassifier("object-detection-yolo/dnn_model/myhaar.xml")
    video = cv2.VideoCapture(path_to_dataset)

    # output video format
    WIDTH = 1280
    HEIGHT = 720

    avg_car_len = 1.8
    fps_list = []
    all_ppms = []

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # %%
    def trackMultipleObjects():
        rectangleColor = (0, 255, 0)
        frameCounter = 0
        currentCarID = 0
        fps = 0

        carTracker = {}
        carLocation1 = {}
        carLocation2 = {}
        speed = [None] * 1000

        # Write output to video file
        out = cv2.VideoWriter(
            "outpy.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (WIDTH, HEIGHT)
        )

        try:

            while video.isOpened():
                start_time = time.time()
                rc, image = video.read()
                if type(image) == type(None):
                    break

                image = cv2.resize(image, (WIDTH, HEIGHT))
                resultImage = image.copy()

                frameCounter = frameCounter + 1

                carIDtoDelete = []

                for carID in carTracker.keys():
                    trackingQuality = carTracker[carID].update(image)

                    if trackingQuality < 7:
                        carIDtoDelete.append(carID)

                for carID in carIDtoDelete:
                    carTracker.pop(carID, None)
                    carLocation1.pop(carID, None)
                    carLocation2.pop(carID, None)

                if not (frameCounter % 10):
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

                    for (_x, _y, _w, _h) in cars:
                        x = int(_x)
                        y = int(_y)
                        w = int(_w)
                        h = int(_h)

                        x_bar = x + 0.5 * w
                        y_bar = y + 0.5 * h

                        matchCarID = None

                        for carID in carTracker.keys():
                            trackedPosition = carTracker[carID].get_position()

                            t_x = int(trackedPosition.left())
                            t_y = int(trackedPosition.top())
                            t_w = int(trackedPosition.width())
                            t_h = int(trackedPosition.height())

                            t_x_bar = t_x + 0.5 * t_w
                            t_y_bar = t_y + 0.5 * t_h

                            if (
                                    (t_x <= x_bar <= (t_x + t_w))
                                    and (t_y <= y_bar <= (t_y + t_h))
                                    and (x <= t_x_bar <= (x + w))
                                    and (y <= t_y_bar <= (y + h))
                            ):
                                matchCarID = carID

                        if matchCarID is None:
                            # print ('Creating new tracker ' + str(currentCarID))

                            tracker = dlib.correlation_tracker()
                            tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                            carTracker[currentCarID] = tracker
                            carLocation1[currentCarID] = [x, y, w, h]

                            currentCarID = currentCarID + 1

                for carID in carTracker.keys():

                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    cv2.rectangle(
                        resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4
                    )
                    # average car width
                    ppm = t_w / avg_car_len
                    print(ppm)
                    all_ppms.append(ppm)

                    if carID >= 5:
                        video.release()
                        out.release()
                        cv2.destroyAllWindows()

                        avg_ppm = (sum(all_ppms) / len(all_ppms))
                        avg_fps = fps
                        print("avg ppm over all:" + str(avg_ppm))
                        print("last fps over all:" + str(avg_fps))

                        return avg_fps, avg_ppm

                    cv2.putText(
                        resultImage,
                        f"{ppm:.2f}ppm",
                        (int(t_x + t_w), int(t_y + t_h)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.95,
                        (255, 255, 255),
                        1,
                    )
                    # speed estimation
                    carLocation2[carID] = [t_x, t_y, t_w, t_h]

                end_time = time.time()

                if not (end_time == start_time):
                    fps = 1.0 / (end_time - start_time)

                if int(major_ver) < 3:
                    fps_global = video.get(cv2.cv.CV_CAP_PROP_FPS)
                    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps_global))
                else:
                    fps_global = video.get(cv2.CAP_PROP_FPS)
                    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps_global))

                fps_list.append(fps)
                cv2.putText(resultImage, 'FPS: ' + str(int(fps_global)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                            (255, 255, 255), 1)
                # cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (7, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)

                for i in carLocation1.keys():
                    if frameCounter % 1 == 0:
                        [x1, y1, w1, h1] = carLocation1[i]
                        [x2, y2, w2, h2] = carLocation2[i]

                        # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                        carLocation1[i] = [x2, y2, w2, h2]

                cv2.imshow("result", resultImage)
                out.write(resultImage)

                if cv2.waitKey(1) == 27 & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            video.release()
            out.release()
            cv2.destroyAllWindows()
            # print("avg ppms over all:")
            # avg_ppm = (sum(all_ppms)/len(all_ppms))
            # avg_fps = sum(fps_list)/(len(fps_list))
            pass

    # %%
    avg_fps, avg_ppm = trackMultipleObjects()
    return avg_fps, avg_ppm


#print(get_fps_and_ppm("../datasets/ori.avi"))
