import cv2

im1 = cv2.imread("image1.jpg", 0)
im2 = cv2.imread("image2.jpg", 0)


im3 = cv2.addWeighted(im1,0.89,im2,0.11,0)
im4  =cv2.addWeighted(im1, 0.477, im2, 0.523, 0)


cv2.imshow("Figure 1", im3)
cv2.imshow("Figure 2", im4)

cv2.waitKey(0); cv2.destroyAllWindows();
