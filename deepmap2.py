import cv2
import numpy as np
import time
from sklearn.preprocessing import normalize
import pandas as pd

# import plotly.graph_objects as go

import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt


def sift(im1, im2):
    start = time.time()
    sift = cv2.SIFT_create()
    keypoints1, des1 = sift.detectAndCompute(im1, None)  # image1 or c1
    keypoints2, des2 = sift.detectAndCompute(im2, None)  # image2 or c2
    # both = np.concatenate((c1, c2), axis=1)
    # initialize Brute force matching
    bestfit = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bestfit.match(des1, des2)
    # sort the matches
    matches = sorted(matches, key=lambda match: match.distance)
    matched_imge = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches[:30], None)
    finish = time.time() -start
    return keypoints1, keypoints2, matches, matched_imge, finish


def orb(im1, im2):
    start = time.time()
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
    finish = time.time() - start
    return  keypoints1, keypoints2, matches, matched_imge, finish
    # Draw top matches


def homo(im1, keypoints1, im2, keypoints2, matches):
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
    height1, width1, channels = im1.shape
    im1Reg = cv2.warpPerspective(im1, matrix, (width, height))
    im2Reg = cv2.warpPerspective(im2, matrix, (width1, height1))
    # creates StereoBm object

    # allframe = np.concatenate((im1Reg,imMatches), axis=1)
    return matrix, im1Reg , im2Reg

def disparity(im1Reg , im2Reg):
    # SGBM Parameters -----------------
    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.4
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(im1Reg, im2Reg)  # .astype(np.float32)/16
    dispr = right_matcher.compute(im2Reg, im1Reg)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, im1Reg, None, dispr)  # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    disparity_SGBM = np.uint8(filteredImg)
    return disparity_SGBM



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
    # preform functions
    orbkeypoints1, orbkeypoints2, orbmatches, orbmat, orbtime = orb(refFilename, imFilename)
    siftkeypoints1, siftkeypoints2, siftmatches, siftmat, sifttime = sift(refFilename, imFilename)

    orbmatrix, orbview_L ,orbview_R  = homo(refFilename, orbkeypoints1, imFilename, orbkeypoints2, orbmatches)
    siftmatrix, siftview_L, siftview_R = homo(refFilename, siftkeypoints1, imFilename, siftkeypoints2, siftmatches)
    # combine input video to output
    inputcat = np.concatenate((refFilename, imFilename), axis=1)
    orbcat = np.concatenate((orbview_L, orbview_R), axis=1)
    siftcat = np.concatenate((siftview_L, siftview_R), axis=1)
    #allframe = np.concatenate((inputcat,orbcat,siftcat), axis=1)

    matchtimes = time.time() - start_time
    print("ORB Time: ", orbtime)
    print("SIFT Time: ", sifttime)
    print("Match Time: ",matchtimes)

    #cv2.imshow("Matching Images", orbcat)
    orbdisparity = disparity(orbview_L, orbview_R)
    siftdisparity = disparity(siftview_L, siftview_R)

    cv2.imshow("SIFT disparity", siftdisparity)
    cv2.imshow("ORB disparity", orbdisparity)
    cv2.imshow("output", inputcat)
    finish_time = time.time() - start_time
    print("Total: ", finish_time)
    #########################
    cv2.imwrite("Experiment/Input/Input_Feed%d.jpg" % count, inputcat)  # save frame as JPEG file
    ###ORB
    cv2.imwrite("Experiment/ORB/Matches/Matches%d.jpg" % count, orbmat)  # save frame as JPEG file
    cv2.imwrite("Experiment/ORB/Homo/Homo%d.jpg" % count, orbcat)  # save frame as JPEG file
    cv2.imwrite("Experiment/ORB/Disparity/Disparity%d.jpg" % count, orbdisparity)  # save frame as JPEG file
    ###SIFT
    cv2.imwrite("Experiment/SIFT/Matches/Matches%d.jpg" % count, siftmat)  # save frame as JPEG file
    cv2.imwrite("Experiment/SIFT/Homo/Homo%d.jpg" % count, siftcat)  # save frame as JPEG file
    cv2.imwrite("Experiment/SIFT/Disparity/Disparity%d.jpg" % count, siftdisparity)  # save frame as JPEG file
    matchtimes = time.time() - start_time
    print("Total: ", matchtimes)
    #########################

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    count += 1
cv2.destroyAllWindows()
