import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Deep_Learning/Self_Organizing_Maps/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""## Feature Scaling"""
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

"""##Visualizing the results"""

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

"""## Finding the frauds"""

mappings = som.win_map(X)
frauds = mappings[(7,1)]
frauds = np.concatenate((mappings[(7,1)], mappings[(6,2)], mappings[(6,3)]), axis = 0)

frauds = sc.inverse_transform(frauds)

"""##Printing the Fraunch Clients"""

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))

#features
customers = dataset.iloc[:,1:].values

#label
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import GridSearchCV

X = customers
y = is_fraud

#data balancing
#smote oversampleing
from imblearn.over_sampling import SMOTE

X_resample, y_resample = SMOTE().fit_resample(X,y.values.ravel())
y_resample = pd.DataFrame(y_resample)
X_resample = pd.DataFrame(X_resample)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN model
model = Sequential([
    Dense(units=16, kernel_initializer= 'uniform', input_dim = 15,activation='relu'),
    Dense(units=24,  activation='relu'),
    Dropout(0.5),
    Dense(units=20, activation='relu'),
    Dense(units=24,activation='relu'),
    Dense(units=26,activation='relu'),
    Dense(units=30,activation='relu'),
    Dense(units=1,activation='sigmoid'),
])

model.summary()

#compile
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=32,epochs=100)

score = model.evaluate(X_test, y_test)
print(score)

y_pred = model.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]