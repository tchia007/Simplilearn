'''
This code has been copied from the simplilearn deep learning tutorial
https://www.youtube.com/watch?v=Jy9-aGMB_TE&index=26&list=PLEiEAq2VkUULYYgj13YHUWmRePqiu8Ddy
data found here: https://www.cs.toronto.edu/~kriz/cifar.html
Problem statement: Using CIFAR-10 dataset for classifying images across 10 categories 
'''


#--------------1. Download data set 
CIFAR_DIR = 'cifar-10-batches-py/'

#--------------2. Import the CIFAR data set 
import pickle 
def unpickle(file):
	with open(file, 'rb') as fo:
		cifar_dict = pickle.load(fo)
	return cifar_dict

dirs = ["batches.meta", "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
all_data = [0,1,2,3,4,5,6]

# print(CIFAR_DIR+direc)
for i, direc in zip(all_data, dirs):
	all_data[i] = unpickle(CIFAR_DIR + direc)


batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

#--------------3. Reading the label names 
print batch_meta

#--------------4. Display the images using matplotlib
import matplotlib.pyplot as plt 
import numpy as np 

X = data_batch1[b"data"]

#reshaping the data. 10k images 32x32 picture 3 bits of color. 
#transpose the color to the last spot 
#0 is the 2 and 3 are the 32 x 32 image 
#need uint8 because its more memory efficient. float is really bad 
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
print X[0].max()
# (X[0] / 255).max()

#just looking at example images inside our dataset 
plt.imshow(X[0]) #imshow = image show only useful in jupyter 
plt.imshow(X[1])
plt.imshow(X[4])

#--------------5. Helper function to handle data 
def one_hot_encode(vec, vals=10):
	'''
	For use to one-hot encode the 10- possible labels 
	'''
	n = len(vec)
	out = np.zeros((n,vals))
	out[range(n), vec] = 1
	return out #output is an array with values of 0 or 1 

class CifarHelper():

	def __init__(self):
		self.i = 0

		self.all_train_batches = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
		self.test_batch = [test_batch]

		#initializing variables to None
		self.training_images = None
		self.training_labels = None
		self.test_images = None
		self.test_labels = None

	def set_up_images(self):

		print "Setting up training images and labels"

		self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
		train_len = len(self.training_images)

		#dividing by 255 because turning the array into a 0-1 range
		self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0,2,3,1)/255
		self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

		print "Setting up test images and labels"

		self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
		test_len = len(self.test_images)

		self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0,2,3,1) / 255

	def next_batch(self, batch_size):
		x = self.training_images[self.i:self.i+batch_size].reshape(100, 32, 32, 3)
		y = self.training_labels[self.i:self.i+batch_size]
		self.i = (self.i + batch_size) % len(self.training_images)
		return x, y

#--------------6. To use the previous code, run the following 
ch = CifarHelper()
ch.set_up_images()

#--------------7. Creating the model
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
hold_prob = tf.placeholder(tf.float32)

#--------------8. Applying the helper functions and create the layers 
#using random numbers to iniaitlize the weights 
#but sometimes you use the same weights to start when trying to compare 2 diff models
def init_weights(shape):
	init_random_dist = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init_random_dist)

def init_bias(shape):
	init_bias_vals = tf.constant(0.1, shape=shape)
	return tf.Variable(init_bias_vals)

#x is data coming in 
#W is filtering 
#strides is like the image where we move our filter over in the larger matrix 
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2b2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], 
						  strides = [1,2,2,1], padding = 'SAME')

def convolutional_layer(input_x, shape):
	W = init_weights(shape)
	b = init_bias([shape[3]])
	return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
	input_size = int(input_layer.get_shape()[1])
	W = init_weights([input_size, size])
	b = init_bias([size])
	return tf.matmul(input_layer, W) + b

#create the layers 
#3 channels 32 pixels each. 4 and 4 are the filter sizes
#can play around with the filter size  
convo_1 = convolutional_layer(x, shape=[4,4,3,32]) 
convo_1_pooling = max_pool_2b2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[4,4,32,64])
convo_2_pooling = max_pool_2b2(convo_2)

#--------------9. Create the flattened layer by reshaping the pooling layer 
convo_2_flat = tf.reshape(convo_2_pooling, [-1,8*8*64])

#--------------10. Create the fully connected layer 
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024)) #1024 is size of layer coming out 

#going backwards to train the data. only retraining a certain %
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob) 

#--------------11. Set output to y_pred
y_pred = normal_full_layer(full_one_dropout, 10) #10 labels
print y_pred

#--------------12. Apply the loss function 
#looking at the average error on this 
#reduce_mean gives us the average. softmax... gives us the error 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))

#--------------13. Create the optimizer 
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cross_entropy)

#--------------14. Create a variable to initialize all the global tf variables 
init = tf.global_variables_initializer()

#--------------15. Run the model by creating a Graph session 
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#going through 500 times. each time is 100 pictures 
	for i in range(500):
		batch = ch.next_batch(100)
		sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})

		#Print message out every 100 steps 
		if i % 100 == 0:

			print("Currently on step {}".format(i))
			print("Accuracy is: ")
			#Test the Train model
			matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

			#average of the accuracy so far 
			acc = tf.reduce_mean(tf.cast(matches, tf.float32))

			#code breaks here
			print(sess.run(acc,feed_dict={x:ch.test_images, y_true:ch.test_labels, hold_prob:1.0}))
			print("\n")






















