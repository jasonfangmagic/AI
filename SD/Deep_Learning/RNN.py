import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the training set
dataset_train = pd.read_csv('Deep_Learning/Part 3 - Recurrent Neural Networks/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling Normalisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timesteps and 1 output
X_train = training_set_scaled[0:999]
y_train = training_set_scaled[1:1000]
X_test = training_set_scaled[1000:]


# Creating a data structure with 60 timesteps and 1 output
# X_train = []
# y_train = []
# for i in range(60, 1258):
#     X_train.append(training_set_scaled[i-60:i, 0])
#     y_train.append(training_set_scaled[i, 0])
# X_train, y_train = np.array(X_train), np.array(y_train)

X_train = X_train.reshape(999,1,1)
X_test = X_test.reshape(258,1,1)

# Reshaping
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1, activation='tanh'))
# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Part 3 - Making the predictions and visualising the results
training_set_pred = regressor.predict(X_train)
training_set_pred = sc.inverse_transform(training_set_pred)
test_set_pred = regressor.predict(X_test)
test_set_pred = sc.inverse_transform(test_set_pred)

#get real result
real_result = sc.inverse_transform(training_set_scaled)
predictons = np.concatenate((training_set_pred,test_set_pred), axis=0)

# # Getting the real stock price of 2017
# dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
# real_stock_price = dataset_test.iloc[:, 1:2].values
#
# # Getting the predicted stock price of 2017
# dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# inputs = inputs.reshape(-1,1)
# inputs = sc.transform(inputs)
# X_test = []
# for i in range(60, 80):
#     X_test.append(inputs[i-60:i, 0])
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# predicted_stock_price = regressor.predict(X_test)
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_result, color = 'red', label = 'Real Google Stock Price')
plt.plot(predictons, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#predict price
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(training_set_scaled[1000:],regressor.predict(X_test)))
print(rmse)