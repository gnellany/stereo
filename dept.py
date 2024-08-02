import cv2
import numpy as np
import time
from sklearn.preprocessing import normalize
import pandas as pd

import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt


def sift(im1, im2):
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
    return im1, keypoints1, im2, keypoints2, matches, matched_imge


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
    return im1, keypoints1, im2, keypoints2, matches, matched_imge
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

    # Visualize the results
    visualize_results(im1, im2, im1Reg, im2Reg)

    return matrix, im1Reg , im2Reg

def disparity(im1Reg , im2Reg):
    # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    block_size = 5
    min_disp = -128
    max_disp = 128
    # Maximum disparity minus minimum disparity. The value is always greater than zero.
    # In the current implementation, this parameter must be divisible by 16.
    num_disp = max_disp - min_disp
    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
    # Normally, a value within the 5-15 range is good enough
    uniquenessRatio = 5
    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleWindowSize = 200
    # Maximum disparity variation within each connected component.
    # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    # Normally, 1 or 2 is good enough.
    speckleRange = 2
    disp12MaxDiff = 0

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )
    disparity_SGBM = stereo.compute(im1Reg , im2Reg)
    plt.imshow(disparity_SGBM, cmap='plasma')
    plt.colorbar()


    # Normalize the values to a range from 0..255 for a grayscale image
    disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=155,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_SGBM = np.uint8(disparity_SGBM)


    return disparity_SGBM

def visualize_results(im1, im2, im1Reg, im2Reg):
    # Display the original and warped images
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Original Image 2')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(im1Reg, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Warped Image 1')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(im2Reg, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Warped Image 2')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


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
    orfim1, orbkeypoints1, orbim2, orbkeypoints2, orbmatches, orbmat = orb(refFilename, imFilename)
    siftim1, siftkeypoints1, siftim2, siftkeypoints2, siftmatches, siftmat = sift(refFilename, imFilename)

    orbmatrix, orbview_L ,orbview_R  = homo(orfim1, orbkeypoints1, orbim2, orbkeypoints2, orbmatches)
    siftmatrix, siftview_L, siftview_R = homo(siftim1, siftkeypoints1, siftim2, siftkeypoints2, siftmatches)
    # combine input video to output
    # inputcat = np.concatenate((refFilename, imFilename), axis=1)
    orbcat = np.concatenate((orbview_L, orbview_R), axis=1)
    # siftcat = np.concatenate((siftview,siftmat), axis=1)
    # allframe = np.concatenate((inputcat,orbcat,siftcat), axis=1)

    finish_time = time.time() - start_time
    print(finish_time)
    orbcat = np.concatenate((orbview_L, orbview_R), axis=1)
    cv2.imshow("Matching Images", orbcat)

    disparity_SGBM = disparity(orbview_L, orbview_R)
    cv2.imshow("Disparity", disparity_SGBM)


    #########################

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    count += 1
cv2.destroyAllWindows()
