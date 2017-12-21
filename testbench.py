import aimgenerator as aim
import cv2
import numpy as np

# feature parameters - ShiTomasi corners
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)

# PyrLK Optical Flow parameters
lk_params = dict( winSize = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10 , 0.03))

# drawing parameters
draw_params = dict( size = 5,
                    bgcolor = (0, 0, 0),
                    colorRangeMin = 0,
                    colorRangeMax = 255,
                    drawInterval = 1,
                    drawDuration = 50)

# flow colors
color = np.random.randint( draw_params['colorRangeMin'], draw_params['colorRangeMax'], (feature_params['maxCorners'],3))

# Read the input video
vidIn = aim.SourceVideo('dancers.mp4')

# Get initial frame
old_frame = vidIn.readFrame(1)
old_gray = aim.frameToGrayscale(old_frame)
old_points = aim.extractFeatures(old_gray, feature_params)

# drawing points
points_list = []

while(vidIn.isOpened()):
    frame = vidIn.readFrame(draw_params['drawInterval'])
    frame_gray = aim.frameToGrayscale(frame)

    # calculate PyrLK optical Flow
    points_new, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)

    # select good points
    good_new = points_new[status==1]
    good_old = old_points[status==1]

    # draw the empty canvas with background color
    canvas = np.zeros_like(old_frame)
    canvas[:] = draw_params['bgcolor']

    # create the flows
    frame_lines = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        frame_lines.append(((a,b),(c,d)))

    # update points_list
    points_list.append(frame_lines)
    if len(points_list)>draw_params['drawDuration']:
        points_list = points_list[1:]

    # draw the flows
    for frame_lines in points_list:
        for i in range(0, len(frame_lines)):
            line = frame_lines[i]
            canvas = cv2.line(canvas, line[0], line[1], color[i].tolist(), 2)
            canvas = cv2.circle(canvas, line[0], draw_params['size'], color[i].tolist(), -1)

    img = cv2.add(frame,canvas)

    cv2.imshow('frame', img)
    cv2.imshow('canvas', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    old_gray = frame_gray.copy()
    old_points = good_new.reshape(-1,1,2)

vidIn.close()
cv2.destroyAllWindows()
