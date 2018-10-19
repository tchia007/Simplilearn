'''
This code has been copied from the simplilearn deep learning tutorial 
https://www.youtube.com/watch?v=_NMI8peAmNA&index=25&list=PLEiEAq2VkUULYYgj13YHUWmRePqiu8Ddy
'''

#--------------1. Import the necessary libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

#--------------2. Read the dataset and print the head of it 
milk = pd.read_csv("monthly-milk-production-pounds-p.csv", index_col = 'Month')
print milk.head()

#--------------3. Convert the index to time series
milk.index = pd.to_datetime(milk.index)

#--------------4. Plot the time series dataset
print milk.plot()

#--------------5. Perform the train test split on the data 
print milk.info()

#we take the 13 years of data for training 
train_set = milk.head(156)

#remaining one year for testing 
test_set = milk.tail(12)

#--------------6. Scale the data using standard machine learning process
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)

#--------------7. Applying the batch function
def next_batch(training_data, batch_size, steps):

	#grab a random starting point for each batch
	rand_start = np.random.randint(0, len(training_data)-steps)

	#create Y data for time series in the batches 
	y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1, steps+1)

	return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)

#--------------8. Setting up the RNN model
import tensorflow as tf 

#just one feature, the time series 
num_inputs = 1 

#num of steps in each batch 
num_time_steps = 12

#100 neuron layer, play with this 
num_neurons = 100

#just one output, predicted time series 
num_outputs = 1

#You can also try increasing iterations, by decreasing learning rate 
#learning rate, can play with this
learning_rate = 0.03 

#how many iterations to go through (training steps), you can play with this 
num_train_iterations = 4000

#size of the batch of data 
batch_size = 1

#--------------9. Create placeholders for X and y
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

#RNN layer 
cell = tf.contrib.rnn.OutputProjectionWrapper(
	   tf.contrib.rnn.GRUCell(num_units = num_neurons, activation = tf.nn.relu), 
	   output_size = num_outputs)

#Pass in the cells variable into tf.nn.dynamic_rnn, along with your first placeholder (X)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)

#--------------10. Applying the loss function and optimizer 
loss = tf.reduce_mean(tf.square(outputs - y)) #MSE 
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)

#--------------11. Initialize the Global variables 
init = tf.global_variables_initializer()

#--------------12. Create an instance of tf.train.Saver()
saver = tf.train.Saver()

#--------------13. Create the session and run it 
with tf.Session() as sess:
	sess.run(init)

	for iteration in range(num_train_iterations):

		 X_batch, y_batch = next_batch(train_scaled, batch_size, num_time_steps)
		 sess.run(train, feed_dict = {X: X_batch, y:y_batch})

		 if iteration % 100 == 0:

		 	mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
		 	print(iteration, "\tMSE:", mse)

	#save model for later 
	saver.save(sess, "./ex_time_series_model")

#--------------14. Display the Test data
print test_set

#--------------15. Create a seed training_instance to predict the last 12 months milk production from the training data 
with tf.Session() as sess:

	#Use your Saver instance to resotre your saved rnn time series model 
	saver.restore(sess, "./ex_time_series_model")

	#Create a numpy array for your generative seed from the last 12 months of the 
	#training set data. Hint: just use tail(12) and then pass it to an np.array 
	train_seed = list(train_scaled[-12:])

	#Now create a for loop for that 
	for iteration in range(12):
		X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
		y_pred = sess.run(outputs, feed_dict = {X: X_batch})
		train_seed.append(y_pred[0, -1, 0])

#--------------16. Displaying the results of the prediction 
print train_seed

#--------------17. Reshape the results 
results = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))

#--------------18. Create the new oclumn on the test data called Generated 
test_set["Generated"] = results

#--------------19. View the test_set dataframe 
print test_set

#--------------20. Plot the predicted result and the actual result 
test_set.plot()






















