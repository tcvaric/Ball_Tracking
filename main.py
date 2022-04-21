import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import socket

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

success, img = cap.read()
h, w, _ = img.shape

myClorFinder = ColorFinder(False)
hsvVals = {'hmin': 40, 'smin': 54, 'vmin': 137, 'hmax': 179, 'smax': 255, 'vmax': 255}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5002)

while True:
    success, img = cap.read()
    if not success:
        break

    imgColor, mask = myClorFinder.update(img, hsvVals)
    imgContour, contours = cvzone.findContours(img, mask)
    if contours:
        data = contours[0]['center'][0], \
               h - contours[0]['center'][1], \
               int(contours[0]['area'])
        print(data)
        sock.sendto(str.encode(str(data)), serverAddressPort)

    #imgStack = cvzone.stackImages([img, imgColor, mask, imgContour], 2, 0.5)
    #cv2.imshow("Image", imgStack)
    #cv2.imshow("ImageColor", imgColor)
    imgContour = cv2.resize(imgContour, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", imgContour)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()