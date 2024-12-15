import cv2

image = cv2.imread('result/image/capture_pose_1.jpg', cv2.IMREAD_GRAYSCALE )
mask_image = cv2.imread('result/binary_image/capture_pose_1.png', cv2.IMREAD_GRAYSCALE)

black = cv2.bitwise_and(image, mask_image)

success = cv2.imwrite("output.jpg", black)

# cv2.imshow('image', black)

# cv2.waitKey(0)