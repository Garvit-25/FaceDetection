import cv2

cascadeClassifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

videoCapture = cv2.VideoCapture(0)
videoCapture.set(3, 720)
videoCapture.set(4, 480)

while True:
    ret, img = videoCapture.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detectedFaces = cascadeClassifier.detectMultiScale(grayImg, scaleFactor=1.2, minNeighbors=7, minSize=(20, 20))

    for (x, y, width, height) in detectedFaces:
        cv2.rectangle(img, (x, y), (x + width, y + height), (128, 128, 0), 3)

    cv2.imshow('Real Time Face Detection', img)

    key = cv2.waitKey(30) & 0xff
    if key == 27:  # If escape key is pressed loop breaks and the window closes
        break

videoCapture.release()
cv2.destroyAllWindows()
