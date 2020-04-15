import os, cv2
import numpy as np

allgray_image = cv2.imread('allgray_image.jpg')


gaussian = np.zeros(allgray_image.shape)
cv2.randn(gaussian,mean=0,stddev=25)
final_image = np.clip(allgray_image+gaussian, 0, 255).astype(np.uint8)
# final_image[:,:,1] = final_image[:,:,0]
# final_image[:,:,2] = final_image[:,:,0]


# noise = np.random.normal(scale=0.01, size=allgray_image.shape)
# final_image = np.clip(allgray_image+noise, 0., 255.).astype(np.uint8)
final_image[:,:,1] = final_image[:,:,0]
final_image[:,:,2] = final_image[:,:,0]
print(final_image)

cv2.imshow('',final_image)
cv2.waitKey(10000)