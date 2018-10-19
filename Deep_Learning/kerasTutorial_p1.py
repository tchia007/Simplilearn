'''
This code is copied for learning purposes from simplilearn at this link
https://www.youtube.com/watch?v=ysVOhBGykxs&index=23&list=PLEiEAq2VkUULYYgj13YHUWmRePqiu8Ddy
'''

#Import the required packages 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN (classifier nueral network)
classifier = Sequential() #sequential is the function call for the neural network model

#--------------Step 1 - Convolution 
#adding first layer 64 x 64 pixels each with 3 values 
#input shape needs to match your data 
#conv2D is conversion 2D. used to convert the photo to a 2 dimensional setup
classifier.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))

#--------------Step 2 Adding a second convolutional layer 
classifier.add(Conv2D(32, (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))#Pooling. changing the dimension to 2D?

#--------------Step 3 Flattening 
classifier.add(Flatten()) #flattening to a single array without multiple dimensions

#--------------Step 4 Full connection
#adding final two dense layers to reduce the layer 
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
#the optimizer is the reverse propogation. adam most used and works well for large data
#optimizer is how the CNN takes the errors and retrains the weights 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metris = ['accuracy'])

#--------------Part 2 Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, #255 is the scale in the colors of the pictures we're using 
								  shear_range = 0.2, 
								  zoom_range = 0.2, 
								  horizontal_flip = True)
training_set = train_datagen.flow_from_directory('path_to_training_set',
												 target_size = (64, 64), #64x64 for the image
												 batch_size = 32, 
												 class_mode = 'binary') #changing everything to binary value

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = train_datagen.flow_from_directory('path_to_test_set',
												 target_size = (64, 64), #64x64 for the image
												 batch_size = 32, 
												 class_mode = 'binary') #changing everything to binary value

#Training the model
#fit generator is the back propogation. essentially training the neural network
classifier.fit_generator(training_set, 
						 steps_per_epoch = 4000, #number of pictures we go through each time 
						 epochs = 10, #number of times we go through each dataset 
						 validation_data = test_set, 
						 validation_steps = 10)

#--------------Part 3 Making new predictions 
import numpy as numpy
from keras.preprocessing import image 

test_image = image.load_img('path_to_test_image.jpg', target_size = (64,64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) #axis = 0 puts it into a single array
result = classifier.predict(test_mage)
training_set.class_indices 
if result[0][0] == 1:
	prediction = 'dog'
else:
	prediction = 'cat'

print prediction











