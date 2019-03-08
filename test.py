import cv2

face_cascade = cv2.CascadeClassifier("C:/Users/4068979/work_nakashima/25_02_2019-SGTC+PSL/haarcascade_frontalface_alt2.xml")
cap = cv2.VideoCapture("C:/Users/4068979/work_nakashima/projects/20190226_new_pd_evaluation/data/movie/sub101_15fps.mp4")

ref, img = cap.read()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

size = int(min(img_gray.shape[1], img_gray.shape[0])*0.15)
faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=1, minSize=(size, size), flags=0)
