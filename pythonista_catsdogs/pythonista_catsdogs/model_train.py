# From https://pythonistaplanet.com/image-classification-using-deep-learning/

from keras.models import Sequential
from keras.layers import Conv2D,Activation,MaxPooling2D,Dense,Flatten,Dropout
import numpy as np

### Create the Model
# Init a CNN using the Sequential (as opposed to Functional) model of keras
classifier = Sequential()

# Add layers
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3)))  # Conv layer: 32 3x3 filters
classifier.add(Activation('relu'))  # Activation layer
classifier.add(MaxPooling2D(pool_size =(2,2)))  # Pooling layer: 2x2 pooling

classifier.add(Conv2D(32,(3,3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

classifier.add(Conv2D(32,(3,3)))  # Note: his example pictured is inconsistent here. Matches if we make this 64,(3,3)
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Dropout to prevent overfitting
classifier.add(Flatten())  # Makes 1-D to prep for dropout
classifier.add(Dense(64))  # Initialize fully connected network w/ ReLU
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(1))  # After dropout, add 1 more fully connected layer. Output n-Dimensional vector
                          # where n=2 (cats, dogs)

classifier.add(Activation('sigmoid')) # Sigmoid activation function will convert the data to probabilities for each class

classifier.summary()  # This is the model

### Compile the Model
# Before training we compile the model
classifier.compile(optimizer ='rmsprop',
                   loss ='binary_crossentropy',
                   metrics =['accuracy'])
                   # 'rmsprop' is the Gradient Descent optimization algorithm
                   # 'binary_crossentropy' is the best loss func for binary classification problems (cats, dogs)
                   # 'accuracy' tells it to give us an accuracy metric

### Data Augmentation
# Data augmentation (increasing the # of images) is done before training to reduce overfitting
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale =1./255,
                                   shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip =True)
test_datagen = ImageDataGenerator(rescale = 1./255)
    # We set these parameters so that the machine will get trained with the images at different positions and
    # details to improve the accuracy

# Set Train and Test directories
print('WARNING: make sure the training set finds EXACTLY "2" classes below')
  # If there are >2 classes, no errors thrown but the results will be wrong.
  #   I was seeing "dog with probability=1.0 for every input"
  # More variations in filenames will cause more "classes" to be found.
  #   flow_from_directory() is particular- you may need to make a sub-sub-directory.
import os
base_dir = os.getcwd()  # Run from Script directory
training_set = train_datagen.flow_from_directory(base_dir+'/train/',
                                                target_size=(64,64),
                                                batch_size= 32,
                                                class_mode='binary')  # subdir train/ implied ands must be omitted

test_set = test_datagen.flow_from_directory(base_dir+'/test/',
                                           target_size = (64,64),
                                           batch_size = 32,
                                           class_mode ='binary')  # subdir test/ implied and must be omitted

### Train the Model
from IPython.display import display
from PIL import Image  # now pillow

# Per the warnings, fit_generator() is deprecated, use fit().
# We will omit giving it a validation data set, since it just slows it down.
#   The correct way to use it, I think, is rather to remove a subset of your training
#   dataset and move them into a test folder, so they are still labeled by class,
#   and the training then gives an unbiased estimate of performace as it trains.
#classifier.fit_generator(training_set,
#                        steps_per_epoch =625,
#                        epochs = 30,
#                        validation_data =test_set,
#                        validation_steps = 5000)
print('Skipping validating test dataset')
classifier.fit(training_set,
        steps_per_epoch =625,
        epochs = 30)
                        # steps_per_epoch and epochs guesses often to be optimized by many trials
                        # Reduce them to speed up training.
# If you get errors on Windows Anaconda here saying:
#   "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5 already initialized." ...
# It's because Anaconda includes certain pre-compiled binaries and they are conflicting with ones we installed, and
#   conda did not catch this. There is a suggested workaround, but it could lead to bad results without errors.
#   Run the following from the environment prompt
#   `conda install nomkl`
#

# See: https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial

# Save the classifier
classifier.save('catdog_cnn_model.h5')
