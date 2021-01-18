import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import keras
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

path = "UTKFace"
save_dir = os.path.join(os.getcwd(), 'age_gender_saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_name = 'age_gender_model.h5'
model_path = os.path.join(save_dir, model_name)
pixels = []
age = []
gender = []
for img in os.listdir(path):
    ages = img.split("_")[0]
    genders = img.split("_")[1]
    img = cv2.imread(str(path)+"/"+str(img))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pixels.append(np.array(img))
    age.append(np.array(ages))
    gender.append(np.array(genders))
age = np.array(age,dtype=np.int64)
pixels = np.array(pixels)
gender = np.array(gender,np.uint64)

x_train,x_test,y_train,y_test = train_test_split(pixels,age,random_state=100)
x_train_2,x_test_2,y_train_2,y_test_2 = train_test_split(pixels,gender,random_state=100)

input = Input(shape=(200,200,3))
conv1 = Conv2D(140,(3,3),activation="relu")(input)
conv2 = Conv2D(130,(3,3),activation="relu")(conv1)
batch1 = BatchNormalization()(conv2)
pool3 = MaxPool2D((2,2))(batch1)
conv3 = Conv2D(120,(3,3),activation="relu")(pool3)
batch2 = BatchNormalization()(conv3)
pool4 = MaxPool2D((2,2))(batch2)
flt = Flatten()(pool4)
#age
age_l = Dense(128,activation="relu")(flt)
age_l = Dense(64,activation="relu")(age_l)
age_l = Dense(32,activation="relu")(age_l)
age_l = Dense(1,activation="relu")(age_l)
#gender
gender_l = Dense(128,activation="relu")(flt)
gender_l = Dense(80,activation="relu")(gender_l)
gender_l = Dense(64,activation="relu")(gender_l)
gender_l = Dense(32,activation="relu")(gender_l)
gender_l = Dropout(0.5)(gender_l)
gender_l = Dense(2,activation="softmax")(gender_l)

model = Model(inputs=input,outputs=[age_l,gender_l])
model.compile(optimizer="adam",loss=["mse","sparse_categorical_crossentropy"],metrics=['mae','accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

checkpoint = ModelCheckpoint(
        filepath= save_dir + "/model-{epoch:02d}-{val_loss:.2f}.h5",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        period=5)

save = model.fit(x_train,[y_train,y_train_2],
    validation_data=(x_test,[y_test,y_test_2]),
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping,checkpoint])

model.save(model_path)
