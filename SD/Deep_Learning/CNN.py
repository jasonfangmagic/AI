import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

# Initialising the CNN
cnn = tf.keras.models.Sequential()
# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=[256, 256, 3]))
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())
# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
# Step 5 - Output Layer
cnn.add(Dropout(0.1))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Data Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('Deep_Learning/Part 2 - Convolutional Neural Networks/dataset/training_set',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'binary')



# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('Deep_Learning/Part 2 - Convolutional Neural Networks/dataset/test_set',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Training the CNN on the Training set and evaluating it on the Test set
cnn.summary()

cnn.fit(x = training_set, validation_data = test_set, epochs = 50)

#save model

model = cnn
model.save('/Users/jasonfang/Work/AI/SD/Deep_Learning/cat_dog')

#load model
cnn = keras.models.load_model('Deep_Learning/cat_dog')


# import pickle
# with open("cat_dog.pickle", "wb") as f:
#     pickle.dump(cnn, f)

# pickle_in = open("cat_dog.pickle", "rb")
#
# cnn_model = pickle.load(pickle_in)

#predict model
from tensorflow.keras.preprocessing import image
test_image = image.load_img('Deep_Learning/Part 2 - Convolutional Neural Networks/dataset/single_prediction/cat1.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)