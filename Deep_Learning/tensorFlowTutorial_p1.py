'''
This code is a copy of the TensorFlow tutorial from this link
https://www.youtube.com/watch?v=skf35x1lNV4&index=21&list=PLEiEAq2VkUULYYgj13YHUWmRePqiu8Ddy
'''


#------------1. import MNIST data using TensorFlow
import numpy as np

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data

#one_hot = true means labeled information is stored as an array
#only one of the digits is labeled. xth position means the labeled value 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#------------2. Check the type of Dataset 
print type(mnist)

#------------3. Array of Training images
print mnist.train.images

#------------4. Number of images for Training, Testing, and Validation
print mnist.train.num_examples
print mnist.test.num_examples
print mnist.validation.num_examples

#------------5. Visualizing the data 
import matplotlib.pyplot as plt

print mnist.train.images[1].shape #gives dimensions of the array
#----code for showing from the youtube video 
plt.imshow(mnist.train.images[1].reshape(28, 28)) #color 
plt.imshow(mnist.train.images[1].reshape(28, 28), cmap='gist_gray') #blk/white

# plt.show()

#------------6. axmimum and minimum value of the pixels in the image
print mnist.train.images[1].max()

#------------7. Create the model 
#placeholder is used to get data from outside the neural network
x = tf.placeholder(tf.float32, shape=[None, 784]) 

#10 because 0-9 numbers
W = tf.Variable(tf.zeros([784, 10])) #W = weights
b = tf.Variable(tf.zeros([10])) #b = biases

#create the graph 
y = tf.matmul(x,W) + b #matmul = matrix multiplication function
y_true = tf.placeholder(tf.float32, [None, 10])

#Cross Entropy 
#reduce_mean is like saying reducing error. cross_entropy is the loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

#using gradient descent optimizer 
#learning rate here is 0.5 but in real world needs to be iteratively tested
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

#------------8. Create the session
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	#Train the model for 1000 steps on the training set using built in batch feeder from mnist
	for step in range(1000):

		#doing training in batches of 100 images here
		#can't do everything because we don't have enough memory to handle all images
		#batch_x = images, batch_y = labels
		batch_x, batch_y = mnist.train.next_batch(100)
		sess.run(train, feed_dict={x:batch_x, y_true:batch_y})

#------------9. Evaluate the Trained model on Test data 
	#Test the trained model 
	matches = tf.equal(tf.argmax(y,1), tf.argmax(y_true, 1))
	acc = tf.reduce_mean(tf.cast(matches, tf.float32))

	print(sess.run(acc,feed_dict={x:mnist.test.images, y_true:mnist.test.labels}))

tf.logging.set_verbosity(old_v)

