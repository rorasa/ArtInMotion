import numpy as np
import cv2

class SourceVideo:
    def __init__(self, filepath):
        self.file = filepath
        self.vidIn = cv2.VideoCapture(self.file)

    def close(self):
        self.vidIn.release()

    def readFrame(self, interval):
        for i in range(0, interval):
            ret, frame = self.vidIn.read()
        return frame

    def isOpened(self):
        return self.vidIn.isOpened()

def frameToGrayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def extractFeatures(frame, feature_params):
    points = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)
    return points
