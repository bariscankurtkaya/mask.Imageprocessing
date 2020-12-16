#Project: Color Detection with OpenCV
#Author: Barışcan KURTKAYA
#Start_date: 21.09.2020

import cv2
import numpy as np

#/Bilgisayardaki Bir Resmi Kullanmak
#img = cv2.imread('images.jfif',0)

#/Resim Göstermek
#cv2.imshow("images", img)
#cv2.waitKey(0)

#/Kameradan Canlı Görüntü Almak
vcap = cv2.VideoCapture(0)

while True:
	ret, img= vcap.read()
	# Videoyu canlı olarak göstermek için
	cv2.imshow("name", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


	#/Resim Kaydetmek
	#cv2.imwrite("PATH/TO/WRITE", img)

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #RGB 2 HSV

	lower = np.array([50,40,35]) 
	upper = np.array([85,255,255]) #Sınırları belirliyoruz.

	mask = cv2.inRange(hsv, lower, upper) #Saptanan rengi beyaz geri kalanını siyah gösterir

	_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #Bulduğumuz 	objenin etrafını kaplayan noktalar oluşturuyoruz.

	#/Bulduğumuz alanların en büyüğünü seçiyoruz böylelikle oluşan küçük hataları ortadan 		kaldırabiliyoruz.
	areas = [cv2.contourArea(c) for c in contours]
	max_index = np.argmax(areas)
	cnt=contours[max_index]

	#/En büyük kontür alanını matristen seçiyoruz
	c = contours[0]
	area = cv2.contourArea(c)

	#/Seçtiğimiz alanın etrafını saran yeşil bir şekil çiziyoruz.
	epsilon = 0.1*cv2.arcLength(cnt,True)
	approx = cv2.approxPolyDP(cnt,epsilon,True)

	"""
	#/Seçtiğimiz alanın etrafına yeşil renkte bir kare çiziyoruz.
	x,y,w,h = cv2.boundingRect(cnt)
	img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

	#/Seçtiğimiz alanın etrafına kırmızı renkte bir dikdörtgen çiziyoruz.
	rect = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	im = cv2.drawContours(im,[box],0,(0,0,255),2)

	#/Seçtiğimiz alanın etrafına yeşil renkte bir daire çiziyoruz.
	(x,y),radius = cv2.minEnclosingCircle(cnt)
	center = (int(x),int(y))
	radius = int(radius)
	img = cv2.circle(img,center,radius,(0,255,0),2)
	"""

cap.release()
cv.destroyAllWindows()

#/Daha çok işaretleme için https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
