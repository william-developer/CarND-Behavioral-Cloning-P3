'''
Model 
1.preprocess data
2.create model
3.train and validate model
4.save the model
5.take a view on the summary of model
'''

import csv
import cv2
import numpy as np

lines = []
#read csv file as a list
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if reader.line_num == 1:  
            continue
        lines.append(line)

images = []
measurements = []
#read image data include center,left,right camera image, and convert BGR to RGB
for line in lines:
    for i in range(3):
        source_path = line[i]
        file_name = source_path.split('/')[-1]
        current_path = '../data/IMG/'+file_name
        imgBGR = cv2.imread(current_path)
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        images.append(imgRGB)
        bias = 0
        if i==1:
            bias = 0.18
        elif i==2:
            bias =-0.2
        measurement = float(line[3])+bias
        measurements.append(measurement)
#augmente images to improve under or over fitting by fliping image
augmented_images,augmented_measurements=[],[]
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Dropout,Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D 
from keras.layers.pooling import MaxPooling2D
#lenet model
def lenet():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((75,25),(0,0))))
    model.add(Conv2D(6, (5, 5),activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, (5, 5),activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model
#improved nvidianet model with dropout layer
def nvidianet():
    model = Sequential()
    #normalized the data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    #crop the useless image infomation,avoid noise,output(65, 320, 3)
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    #Convolutional feature map 24@31*158
    model.add(Conv2D(24,(5, 5), strides=(2,2) , activation="relu"))
    #Convolutional feature map 36@14*77
    model.add(Conv2D(36,(5, 5), strides=(2,2) , activation="relu"))
    #Convolutional feature map 48@5*37
    model.add(Conv2D(48,(5, 5), strides=(2,2) , activation="relu"))
    #Convolutional feature map 64@3*35
    model.add(Conv2D(64,(3, 3), activation="relu"))
    #Convolutional feature map 64@1*33
    model.add(Conv2D(64,(3, 3), activation="relu"))
    #ouput(2112)
    model.add(Flatten())
    #output(2112)
    model.add(Dropout(0.5))
    #output(100)
    model.add(Dense(100))
    #output(50)
    model.add(Dense(50))
    #output(10)
    model.add(Dense(10))
    #output(1)
    model.add(Dense(1))
    return model
#create the model
model = nvidianet()
#used an adam optimizer so that manually training the learning rate wasn't necessary
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=5,batch_size=32)

#save the model
model.save('model.h5')

#from keras.models import load_model
#from keras.utils import plot_model
#model = load_model('model.h5')
#the summary of the architecture
#print(model.summary())
#a visualization of the architecture
#plot_model(model, to_file='examples/model.png')
exit()
