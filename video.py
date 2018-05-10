import cv2
import numpy as np
from PIL import Image

FRONT_PATH = 'models/haarcascade_frontalface_default.xml'

def create_rectangle(faces, frame):
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 50, 100), 2)

def crop(faces, frame, i):
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cropped = frame[ y: y + h, x: x + w]
        cv2.imshow("cropped", cropped)
        im = Image.fromarray(cropped)
        im.save("cropped" + str(i) + ".png")


def capture_video():
    faceCascade = cv2.CascadeClassifier(FRONT_PATH)
    # creating a face cascade
    video_capture = cv2.VideoCapture(0)
    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, counter = faceCascade.detectMultiScale2(gray, 1.3, 5)
        crop(faces, frame, i)
        #cv2.imshow("cropped", cropped)
        i += 1
        create_rectangle(faces, frame)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    capture_video()

if __name__ == "__main__":
    main()