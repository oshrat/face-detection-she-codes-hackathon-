import cv2
import sys
import numpy as np

FRONT_PATH = 'models/haarcascade_frontalface_default.xml'
SMILE_PATH = 'models/haarcascade_smile.xml'
smileCascade = cv2.CascadeClassifier(SMILE_PATH)
faceCascade = cv2.CascadeClassifier(FRONT_PATH)

def create_rectangle(faces, frame):
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 50, 100), 2)

def crop(faces, frame):
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cropped = frame[ y: y + h, x: x + w]
        cv2.imshow("cropped", cropped)
        return cropped

def capture_video():
    # creating a face cascade
    video_capture = cv2.VideoCapture(0)
    pics = []
    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, counter = faceCascade.detectMultiScale2(gray, 1.3, 5)
        crop(faces, frame)
        #cv2.imshow("cropped", cropped)
        #pics[i] = crop(faces, frame)
        #i += 1
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