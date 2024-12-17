import cv2

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

cv2.imshow("show", frame)
cv2.imwrite("show.jpg", frame)

cv2.waitKey(0)