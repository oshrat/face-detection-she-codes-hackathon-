import cv2
#import sys
import numpy as np
import pandas as pd
from PIL import Image
import os, csv

FRONT_PATH = 'models/haarcascade_frontalface_default.xml'
SMILE_PATH = 'models/haarcascade_smile.xml'
smileCascade = cv2.CascadeClassifier(SMILE_PATH)
faceCascade = cv2.CascadeClassifier(FRONT_PATH)


list_of_names = ["Avigail","Lena","Oshrat"]


def create_rectangle(faces, frame,recognize_face):
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 50, 100), 2)
        #print text in label
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        label = predict_func(gray,recognize_face)
        print("GOOD")
        if label[0] == 1:
            print("11111111")
            cv2.putText(frame, list_of_names[0], (x+2, y+2), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        if label[0] == 2:
            print("22222")
            cv2.putText(frame, list_of_names[1], (x+2, y+2), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)  
        if label[0] == 3:
            print("33333")
            cv2.putText(frame, list_of_names[2], (x+2, y+2), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)  
       


def crop(faces, frame):
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cropped = frame[ y: y + h, x: x + w]
        cv2.imshow("cropped", cropped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return cropped
def capture_video(recognize_face):
    # creating a face cascade
    video_capture = cv2.VideoCapture(0)
   
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, counter = faceCascade.detectMultiScale2(gray, 1.3, 5)
        #crop(faces, frame)
        #cv2.imshow("cropped", cropped)
        #pics[i] = crop(faces, frame)
        #i += 1
        create_rectangle(faces, frame,recognize_face)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def csv_list(path):
    
    print("start")
    f=open(path +"\\train.csv",'w')
    w=csv.writer(f)
    print("Open file........")
    w.writerow(['Foto','Label'])
    for path, dirs, files in os.walk(path):
        for filename in files:
            if filename[0].isdigit():
                w.writerow([filename,filename[0]])
                print("save to file ")
def train_process(path) :
   
    df = pd.read_csv(path +"\\train.csv")
    list_of_faces = df['Foto']
  
    print(list_of_faces)
    print(type(list_of_faces[0]))
    labels = df['Label']
    print(labels)
    
    faces = []
    for image in list_of_faces:
        im = np.asarray(Image.open(path+"\\"+image))
        im_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        print(type(im_gray))
        print(np.shape(im_gray))
        faces.append(im_gray)
        #print(type(np.asarray(Image.open(path+"\\"+image))))
        #faces = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
        #faces = np.asarray(faces)
    
    #create our LBPH face recognizer 
    #recognize_face = cv2.face.createLBPHFaceRecognizer()
    recognize_face = cv2.face_LBPHFaceRecognizer.create()
    #recognize_face = cv2.face_FisherFaceRecognizer.create()      

    #recognize_face= cv2.face_EigenFaceRecognizer.create()
    recognize_face.train(faces, np.array(labels))
    print("recognize :",recognize_face)
    return recognize_face
    
    
def predict_func(face,recognize_face):

     
    #predict the image using our face recognizer 
    label= recognize_face.predict(face)

    print("label: ",label)
    return label






def main():
    path = r"C:\Users\lena_\OneDrive\Desktop\shecodes_hakaton\train_data"
    csv_list(path)
    recognize_face = train_process(path)
    capture_video(recognize_face)
    #create csv file for training
   
   
    
    
    
    
if __name__ == "__main__":
    main()