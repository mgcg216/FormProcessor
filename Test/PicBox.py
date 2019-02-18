import cv2

x = 100
y = 200
h = 300
w = 300
img = cv2.imread("test.jpg")
cv2.imshow("base", img)
cv2.waitKey(0)
crop_img = img[y:y+h, x:x+w]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
