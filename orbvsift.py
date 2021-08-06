import cv2
import numpy as np
import time
import pandas as pd

#import plotly.graph_objects as go

import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt



def sift(im1,im2):
    sift = cv2.SIFT_create()
    keypoints1, des1= sift.detectAndCompute(im1, None)#image1 or c1
    keypoints2, des2= sift.detectAndCompute(im2, None)#image2 or c2
    #both = np.concatenate((c1, c2), axis=1)
    # initialize Brute force matching
    bestfit = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bestfit.match(des1,des2)
    #sort the matches
    matches = sorted(matches, key= lambda match : match.distance)
    matched_imge = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches[:30], None)
    return im1, keypoints1, im2, keypoints2,matches,matched_imge

def orb(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by scoreq
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    matched_imge = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches[:30], None)
    return im1, keypoints1, im2, keypoints2,matches,matched_imge
    # Draw top matches
def homo(im1, keypoints1, im2, keypoints2,matches):
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, matrix, (width, height))

    # creates StereoBm object

    #allframe = np.concatenate((im1Reg,imMatches), axis=1)
    return matrix, im1Reg

# camera feed
video1 = cv2.VideoCapture(1)
video2 = cv2.VideoCapture(0)

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.5
count = 0


while True:
    start_time = time.time()
    # read caqqqmera to while loop
    ret1, refFilename = video1.read()
    ret2, imFilename = video2.read()
    #preform functions
    orfim1, orbkeypoints1, orbim2, orbkeypoints2, orbmatches,orbmat = orb(refFilename, imFilename)
    siftim1, siftkeypoints1, siftim2, siftkeypoints2, siftmatches,siftmat = sift(refFilename, imFilename)

    orbmatrix, orbview = homo(orfim1, orbkeypoints1, orbim2, orbkeypoints2, orbmatches)
    siftmatrix, siftview = homo(siftim1, siftkeypoints1, siftim2, siftkeypoints2, siftmatches)
    #combine input video to output
    #inputcat = np.concatenate((refFilename, imFilename), axis=1)
    orbcat = np.concatenate((orbview,orbmat), axis=1)
    #siftcat = np.concatenate((siftview,siftmat), axis=1)
    #allframe = np.concatenate((inputcat,orbcat,siftcat), axis=1)
    #cv2.imwrite("Seperated_files/frame_left%d.jpg" % count, refFilename)  # save frame as JPEG file
    #cv2.imwrite("Seperated_files/frame_right%d.jpg" % count, imFilename)  # save frame as JPEG file
    #cv2.imwrite("Seperated_files/frame_combine%d.jpg" % count, imReg)  # save frame as JPEG file
    #show data
    finish_time = time.time() - start_time
    print(finish_time)
    cv2.imshow("Matching Images", orbcat)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    count += 1
cv2.destroyAllWindows()
