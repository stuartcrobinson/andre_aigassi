import cv2
from satyaHomography2018 import alignHomography2018
from affineMusic import alignAffine

# import satyaStabilization.align.alignImages as asdf

# Read input video
cap = cv2.VideoCapture('video.mp4')

# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frameIndex1 = 150
frameIndex2 = frameIndex1 + 1

frames = []

for i in range(n_frames - 2):
    success, frame = cap.read()
    #
    if i == frameIndex1 or i == frameIndex2:
        frames.append(frame)
    if i > frameIndex2:
        break

img1 = frames[0]
img2 = frames[1]

imgAlignedHomo, h = alignHomography2018(img2, img1)
# imgAlignedAffine = alignAffine(img2, img1)

cv2.imwrite('img1.jpg', img1)
cv2.imwrite('img2.jpg', img2)
cv2.imwrite('imgH.jpg', imgAlignedHomo)
# cv2.imwrite('imgA.jpg', imgAlignedAffine)
cap = cv2.VideoCapture('video.mp4')


# Release video
cap.release()
# out.release()
# Close windows
cv2.destroyAllWindows()
