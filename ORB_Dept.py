import cv2
import numpy as np
import time

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
    return im1, keypoints1, im2, keypoints2, matches, matched_imge,finish
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
    # Use homography
    height, width, channels = im2.shape
    height1, width1, channels = im1.shape
    im1Reg = cv2.warpPerspective(im1, matrix, (width, height))
    im2Reg = cv2.warpPerspective(im2, matrix, (width1, height1))
    # creates StereoBm object

    # allframe = np.concatenate((im1Reg,imMatches), axis=1)
    return matrix, im1Reg, im2Reg

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
    sigma = 2.1
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



listofitems = []
# camera feed
video1 = cv2.VideoCapture(1)
video2 = cv2.VideoCapture(0)

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.5
count = 0
MATCH_TIME = []
FINISH_TIME = []
while True:
    start_time = time.time()
    # read camera to while loop
    ret1, refFilename = video1.read()
    ret2, imFilename = video2.read()
    # preform functions
    im1, keypoints1, im2, keypoints2, matches, mat, pro_time = orb(refFilename, imFilename)
    matrix, view_L ,view_R  = homo(im1, keypoints1, im2, keypoints2, matches)
    #combine input video to output
    #outputcat = np.concatenate((view_L,view_R), axis=1)
    #orbcat = np.concatenate((orbview_L, orbview_R), axis=1)
    matchtimes = time.time() - start_time
    print("Process Time: ", pro_time)
    print("Match Time: ",matchtimes)
    #cv2.imshow("Matching Images", orbcat)
    #disparity_SGBM = disparity(view_L, view_R)
    #output = np.concatenate((refFilename, imFilename), axis=1)
    #cv2.imshow("outputcat", outputcat)
    #cv2.imshow("output", output)
    finish_time = time.time() - start_time
    print("Total: ", finish_time)
    #########################
    #cv2.imwrite("Seperated_files/disparity%d.jpg" % count, disparity_SGBM)  # save frame as JPEG file
    #cv2.imwrite("Seperated_files/output%d.jpg" % count, output)  # save frame as JPEG file
    #########################
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if count > 60:
        break
    count += 1
cv2.destroyAllWindows()
