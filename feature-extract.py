import numpy as np
import cv2

# feature parameters - ShiTomasi corners
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)

# PyrLK Optical Flow parameters
lk_params = dict( winSize = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10 , 0.03))

# Flow colors
color = np.random.randint(0, 255, (100,3))

# Read the input video
vidIn = cv2.VideoCapture('dancers.mp4')

# Get initial frame
ret, old_frame = vidIn.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
points_old = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# drawing canvas
canvas = np.zeros_like(old_frame)

while(vidIn.isOpened()):
    ret, frame = vidIn.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate PyrLK optical Flow
    points_new, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, points_old, None, **lk_params)

    # select good points
    good_new = points_new[status==1]
    good_old = points_old[status==1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        canvas = cv2.line(canvas, (a,b), (c,d), color[i].tolist(), 2)
        canvas = cv2.circle(canvas, (a,b), 5, color[i].tolist(), -1)
    img = cv2.add(frame,canvas)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    old_gray = frame_gray.copy()
    points_old = good_new.reshape(-1,1,2)

vidIn.release()
cv2.destroyAllWindows()
