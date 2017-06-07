import numpy as np
import cv2

import sys

X = np.load(sys.argv[1])
for x in X:
    cv2.imshow('X', x)
    cv2.waitKey(0)
