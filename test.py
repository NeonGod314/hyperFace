import cv2
import numpy as np
import glob
#data = np.load('hyf_data.npy')
#print(data.shape)

i = 0
files = glob.glob('./dape/*.jpg')
print(files)
for imgs in files:
	#print(imgs[0].shape)
	img = cv2.imread(imgs)
	img = cv2.resize(img,(227,227))
	print(img.shape)
	i = i+1
	#img = cv2.i(imgs[0])
	cv2.imwrite('resized_data/'+'a_'+str(i)+'.jpg', img)