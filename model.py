import csv
import cv2
import os.path
import numpy as np
import sklearn
import math
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D

#----------------------------------------------------------------------------#
#                            Image Import Pipeline                           #
#----------------------------------------------------------------------------#

def get_training_data(image_log, training_data_path, batch_size=22):

  def import_and_convert_RGB(image_file):
    image = cv2.imread(image_file, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image

  def get_training_set(log, image_index, measurement_adjustment=0.0, flip=False):
    images = []
    measurements = []
    for row in log:
      image_name = row[image_index].split('/')[-1]
      image_path = training_data_path + 'IMG/' + image_name
      if os.path.isfile(image_path):
        image = import_and_convert_RGB(image_path)
        measurement = float(row[3]) + measurement_adjustment
        if flip:
          measurement = -measurement
          image = np.fliplr(image)

          
        measurements.append(measurement)
        images.append(image)

    return images, measurements

  #  -------------------------- Main Function Body --------------------------- #

  correction = 0.4 #Angle correction for off center cameras
  num_images = len(image_log)

  while 1: # Do not let the generator terminate
    for offset in range(0, num_images, batch_size):
            batch_samples = image_log[offset:offset+batch_size]

            all_images       = []
            all_measurements = []

            use_data                = [True,  False, False, False, False, False]
            log_indices             = [0,     1,     2,     0,     1,     2]
            flip_states             = [False, False, False, True,  True,  True]
            measurement_adjustments = [0,     1,     -1,    0,     1,     -1]
            measurement_adjustments = /
              [x * correction for x in measurement_adjustments]

            for index, log_index in enumerate(log_indices):
              if use_data[index]:
                images, measurements = get_training_set(
                  batch_samples, log_index, 
                  measurement_adjustment=measurement_adjustments[index], 
                  flip=flip_states[index])

                all_images.extend(images)
                all_measurements.extend(measurements)

            # trim image to only see section with road
            X_train = np.array(all_images)
            y_train = np.array(all_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

#----------------------------------------------------------------------------#
#                          Set up Model and Training                         #
#----------------------------------------------------------------------------#

def test_nn_model(model):
  model.add(Flatten())
  model.add(Dense(1))

  return model

def image_preprocessing(model, image_resolution):
  model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320)))
  model.add(
    Lambda(lambda x: x/255.0 - 0.5, 
    input_shape=(image_resolution[0], image_resolution[1], 3)))
  

  return model

#----------------------------------------------------------------------------#
#                              Main Function Body                            #
#----------------------------------------------------------------------------#

# Import images
rows = []
training_data_path = './data/training_set/'

with open(training_data_path  + 'driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  rows = [row for row in reader]

train_samples, validation_samples = train_test_split(rows, test_size=0.2)

batch_size = 22
n_batches = math.ceil(len(train_samples)/batch_size)
n_val_batches = math.ceil(len(validation_samples)/batch_size)

train_generator = get_training_data(train_samples, training_data_path, batch_size=22)
validation_generator = get_training_data(validation_samples, training_data_path, batch_size=22)

# Set up training pipeline
cur_model = Sequential()

cur_model = image_preprocessing(cur_model, image_resolution=[160,320])
cur_model = test_nn_model(cur_model)
cur_model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, steps_per_epoch=n_batches, 
                    validation_data=validation_generator,
                    validation_steps=n_val_batches, epochs=7, verbose=1)

cur_model.save('model.h5')