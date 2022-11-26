import cv2
import numpy as np

vidcap = cv2.VideoCapture('../datasets/ori.avi')
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(f"frames/frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    count += 1

    if count == 200:
        break
print("I read " + str(count) + " frames!")

for i in range(0, count):

    img = cv2.imread(f'frames/frame%d.jpg' % i)

    pts1 = np.float32([[4, 61], [191, 61], [406, 450], [820, 250]]) #brno comp
    #pts1 = np.float32([[500, 260], [780, 260], [2, 430], [1270, 430]])  # set automatically the street up left, up right, down left, down right
    #above this for youtube_video1

    #below for yt video 2
    #pts1 = np.float32([[530, 290], [730, 290], [5, 610], [1260, 610]])  # set automatically the street up left, up right, down left, down right

    list_x = []
    list_y = []
    for x in range(0, 4):
        list_x.append(pts1[x][0])
        list_y.append(pts1[x][1])

    width, height = max(list_x), max(list_y) #set this bc otherwise pictures different size i guess
    # get highest value from set x and y in picture to not have black spots i think
    # print(width)

    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # set automatically w and h
    # print(pts1)

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    output = cv2.warpPerspective(img, matrix, (int(width), int(height)))

    #for i in range(0, 4):
    #    cv2.circle(img, (int(pts1[i][0]), int(pts1[i][1])), 5, (0, 0, 255), cv2.FILLED)


#cv2.imshow("lol", output)
#cv2.waitKey(0)


    cv2.imwrite(f"converted_frames/birdframe%d.jpg" % i, output)  # save frame as JPEG file


img_array = []
for count in range(0, 800):
    img = cv2.imread(f'converted_frames/birdframe%d.jpg' % count)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('outputs/output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, size) #fps have to get set automatically from orignal video

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
