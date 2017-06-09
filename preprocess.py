import cv2
import sys
import numpy as np

def pre_process(imgs, mean, scale):
    imgs -= mean
    imgs *= scale
    return imgs

def post_process(imgs, mean, scale):
    imgs /= scale
    imgs += mean
    return imgs


class FaceDetector(object):
    def __init__(self, cascPath="./haarcascade_frontalface_default.xml"):
        self.model = cv2.CascadeClassifier(cascPath)

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces, _, scores = self.model.detectMultiScale3(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(25, 25),
            outputRejectLevels = True
        )    
        if type(faces) is tuple:
            return img
        H, W = gray.shape

        inds = np.argsort(scores, axis=0)[::-1]
        faces = faces[inds]
        scores = scores[inds]
        x, y, w, h = faces[0][0]
        x1 = int(np.clip(x       - w*0.2, 0, W))
        x2 = int(np.clip((x + w) + w*0.2, 0, W))
        y1 = int(np.clip(y       - h*0.2, 0, H))
        y2 = int(np.clip((y + h) + h*0.2, 0, H))
        '''
        for i in range(len(faces)):
            face = faces[i]
            x, y, w, h = face[0]
            cv2.imshow('img%d' % i, img[y:y+h, x:x+w])
        cv2.waitKey(0)
        '''
        
        return img[y1:y2, x1:x2]

