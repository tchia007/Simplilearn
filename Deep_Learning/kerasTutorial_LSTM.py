'''
This code has been copied from the simplilearn deep learning tutorial 
https://www.youtube.com/watch?v=lWkFhVq9-nc&index=29&list=PLEiEAq2VkUULYYgj13YHUWmRePqiu8Ddy
Predict stock prices using LSTM network based on stock price data between 2012-2016
'''

#--------------1. Import the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

#--------------2. Import the training dataset 
dataset_train = pd.read_csv("Google_Stock_Price_Trian.csv")
training_set = dataset_train.iloc[:, 1:2].values 

#--------------3. Feature Scaling 
from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#--------------4. Create a data structure with 60 timesteps and 1 output 
X_train = []
y_train = []

#
for i in range(60, 1258):
	#0 is for the opening volumn data 
	X_train.append(training_set_scaled[i-60:i, 0])
	y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#--------------5. Import keras libraries and packages 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout 

#--------------6. Initialize the RNN 
regressor = Sequential()

#--------------7. Adding the LSTM layers and some Dropout regularization 
#Adding first LSTM layer
#units is the positive integer. dimensionality of the output space. whats going out into the next layer
#60 in and 50 out 
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

#dropout beacuse maybe we could be overfitting our data. take 20% of the neurons and turn it off 
regressor.add(Dropout(0.2))

#Adding second LSTM layer 
#don't need the shape because it automatically trickles down 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding third LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding fourth LSTM layer 
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#--------------8. Adding the output layer 
#dense means we're gonna bring this out to 1 output. no sequence. 
regressor.add(Dense(units = 1))

#--------------9. Compile the RNN 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#--------------10. Fit the RNN to the training set 
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#--------------11. Load the stock price test data for 2017
dataset_test = pd.read_csv('test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#--------------12. Get the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test["Open"]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
	X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)	
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#--------------13. Visualize the results of predicted and real stock price 
plt.plot(real_stock_price, color ="red", label = "Real Google Stock Price")
plt.plot(predicted_stock_price, color = "blue", label = "Predicted Google Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()



























