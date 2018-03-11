
# coding: utf-8

# # Read files 
# 

# In[1]:


'''
import csv
import cv2
import numpy as np
lines = []

#csv_file = '../data/driving_log.csv'
csv_file = '../data_2/driving_log.csv'

with open(csv_file ) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    for i in range(3):#left center right camera
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = '../data_2/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

#X_train = np.array(images)
#y_train = np.array(measurements)
'''


# # data augmentation

# In[2]:


def data_aug(image, measurement):
    augmented_images = []
    augmented_measurements = []
    #for image, measurement in zip (images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    return augmented_images, augmented_measurements

#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)


# In[12]:


import os
import csv
import numpy as np


samples = []
csv_file = '../data_2/driving_log.csv'

with open(csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


import sklearn
import csv
import cv2
from random import shuffle

#img_new_shape = (80,160,3)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_c = '../data_2/IMG/'+batch_sample[0].split('/')[-1]#center
                name_l = '../data_2/IMG/'+batch_sample[1].split('/')[-1]#left
                name_r = '../data_2/IMG/'+batch_sample[2].split('/')[-1]#right
                
                center_image = cv2.imread(name_c)
                left_image = cv2.imread(name_l)
                right_image = cv2.imread(name_r)
                 
                #Reorder BGR to RGB
                #CV2 import BGR but we infere the steering in RGB
                center_image = center_image[:, :, (2, 1, 0)]                                
                left_image = left_image[:, :, (2, 1, 0)]                                
                right_image = right_image[:, :, (2, 1, 0)]                                
                
                
                correction = 0.2 # this is a parameter to tune
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                aug_center_image  = []
                aug_center_angle = []
                aug_right_image  = []
                aug_right_angle = []
                aug_left_image  = []
                aug_left_angle = []

                
                aug_center_image, aug_center_angle = data_aug(center_image, center_angle)
                aug_right_image, aug_right_angle = data_aug(right_image, right_angle)
                aug_left_image, aug_left_angle = data_aug(left_image, left_angle)
                #aug_center_image = cv2.flip(center_image,1)
                #aug_center_angle = center_angle*-1.0
                
                #images.append(center_image)
                #angles.append(center_angle)
                
                #images.append(aug_center_image)
                #angles.append(aug_center_angle)

                
                # add images and angles to data set
                images.append(left_image)
                images.append(cv2.flip(left_image,1))
                
                images.append( center_image)
                images.append(cv2.flip(center_image,1))
                
                images.append(right_image)
                images.append(cv2.flip(right_image,1))
                
                
                angles.append(left_angle)
                angles.append(left_angle*-1.0)
                
                angles.append( center_angle)
                angles.append( center_angle*-1.0)
                
                angles.append(right_angle)
                angles.append(right_angle*-1.0)
                
                #images.extend(aug_left_image, aug_center_image, aug_right_image)
                #angles.extend(aug_left_angle, aug_center_angle, aug_right_angle)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#X_train_shuffle = np.zeros(160*320*3*32)
#y_train_shuffle = np.zeros(160*320*3*32)

#X_train_shuffle, y_train_shuffle = train_generator

#print (X_train_shuffle)


# # Generator

# In[13]:


'''
import os
import csv
import numpy as np


samples = []
csv_file = '../data_2/driving_log.csv'

with open(csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


import sklearn
import csv
import cv2
from random import shuffle

#img_new_shape = (80,160,3)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../data_2/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                
                #Reorder BGR to RGB
                #CV2 import BGR but we infere the steering in RGB
                center_image = center_image[:, :, (2, 1, 0)]
                
                #downsample the picture
                #input_shape=(160,320,3)
                #center_image = cv2.resize(center_image, img_new_shape)
                
                center_angle = float(batch_sample[3])
                #real batch_size = batch_size * 2
               # center_image, center_angle = data_aug(center_image, center_angle)

            
                #aug_center_image = []
                #aug_center_angle = []
                #aug_center_image, aug_center_angle =  data_aug(center_image, center_angle)
                #print(aug_center_image)
                
                #images.append(center_image)
                #angles.append(center_angle)
                
                aug_center_image = cv2.flip(center_image,1)
                aug_center_angle = center_angle*-1.0
                
                images.append(center_image)
                angles.append(center_angle)
                
                images.append(aug_center_image)
                angles.append(aug_center_angle)
                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#X_train_shuffle = np.zeros(160*320*3*32)
#y_train_shuffle = np.zeros(160*320*3*32)

#X_train_shuffle, y_train_shuffle = train_generator

#print (X_train_shuffle)
'''


# # data augmentation

# In[14]:


'''
augmented_images, augmented_measurements = [], []
for image, measurement in zip (images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
'''


# # build model 

# In[15]:


import keras
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/ 255.0 - 0.5,  input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
#model.add(Convolution2D(6,5,5,activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
          
model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch=5)


history_object = model.fit_generator(train_generator, samples_per_epoch= 6*len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=6*len(validation_samples), 
                    nb_epoch=3)


model.save('model.h5')          


#    # plot MSE loss

# In[ ]:


import matplotlib.pyplot as plt

#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validation_data = 
#    validation_generator,
#    nb_val_samples = len(validation_samples), 
#    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

