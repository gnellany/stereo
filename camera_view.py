import cv2
import numpy as np

# camera feed
video1 = cv2.VideoCapture(1)
video2 = cv2.VideoCapture(0)

while True:
    # read camera to while loop
    ret1, refFilename = video1.read()
    ret2, imFilename = video2.read()
    allframe = np.concatenate((refFilename, imFilename), axis=1)
    cv2.imshow("Matching Images", allframe)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()