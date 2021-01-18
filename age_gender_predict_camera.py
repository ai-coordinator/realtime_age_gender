# -*- coding: utf-8 -*-
import cv2
import numpy as np
from keras.models import load_model
import cv2
model_path = "./age_gender_saved_models/age_gender_model.h5"
model = load_model(model_path)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

age_ = []
gender_ = []

image_size = 200

camera = cv2.VideoCapture(0)

while True:
        #映像を取得する
        ret, frame = camera.read()
        faces = face_cascade.detectMultiScale(frame,scaleFactor=1.11, minNeighbors=8)

        for (x,y,w,h) in faces:
          img = frame[y:y + h, x:x + w]
          img = cv2.resize(img,(image_size,image_size))
          predict = model.predict(np.array(img).reshape(-1,image_size,image_size,3))
          age_.append(predict[0])
          gender_.append(np.argmax(predict[1]))
          gend = np.argmax(predict[1])
          if gend == 0:
            gend = 'Man'
            col = (255,255,0)
          else:
            gend = 'Woman'
            col = (203,12,255)
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,225,0),1)
          cv2.putText(frame,"Age:"+str(int(predict[0]))+"/"+str(gend),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,w*0.005,col,1)
        cv2.imshow("camera",frame)

        key = cv2.waitKey(10)
        # Escキーが押されたら
        if key == 27:
            cv2.destroyAllWindows()
            break
