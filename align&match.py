import cv2
import numpy as np

def shift(img1,img2):
    sift = cv2.SIFT_create()
    keypoints1, des1= sift.detectAndCompute(img1, None)#image1 or c1
    keypoints2, des2= sift.detectAndCompute(img2, None)#image2 or c2

    #both = np.concatenate((c1, c2), axis=1)

    # initialize Brute force matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1,des2)
    #sort the matches
    matches = sorted(matches, key= lambda match : match.distance)
    matched_imge = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:30], None)
    return matched_imge



def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    ###########
    # #initialize SIFT object
    # sift = cv2.SIFT_create()
    # keypoints1, des1= sift.detectAndCompute(im1Gray, None)#image1 or c1
    # keypoints2, des2= sift.detectAndCompute(im2Gray, None)#image2 or c2
    #
    # #both = np.concatenate((c1, c2), axis=1)
    #
    # # initialize Brute force matching
    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    # matches = bf.match(des1,des2)
    # #sort the matches
    # matches = sorted(matches, key= lambda match : match.distance)
    # matched_imge = cv2.drawMatches(im1Gray, keypoints1, im2Gray, keypoints2, matches[:30], None)
    # ##########


    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    allframe = np.concatenate((im1Reg,imMatches), axis=1)
    return allframe

# camera feed
video1 = cv2.VideoCapture(1)
video2 = cv2.VideoCapture(0)

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.5
count = 0


while True:
    # read camera to while loop
    ret1, refFilename = video1.read()
    ret2, imFilename = video2.read()
    #preform functions
    imReg = alignImages(refFilename, imFilename)
    #combine input video to output
    allframe = np.concatenate((refFilename, imFilename,imReg), axis=1)
    cv2.imwrite("Seperated_files/frame_left%d.jpg" % count, refFilename)  # save frame as JPEG file
    cv2.imwrite("Seperated_files/frame_right%d.jpg" % count, imFilename)  # save frame as JPEG file
    cv2.imwrite("Seperated_files/frame_combine%d.jpg" % count, imReg)  # save frame as JPEG file
    #show data
    cv2.imshow("Matching Images", allframe)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    count += 1
cv2.destroyAllWindows()
