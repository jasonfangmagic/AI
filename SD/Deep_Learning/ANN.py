import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import GridSearchCV
tf.__version__

# Importing the dataset
dataset = pd.read_csv('Deep_Learning/Part 1 - Artificial Neural Networks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

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
    Dense(units=16, init = 'uniform', input_dim = 12,activation='relu'),
    Dense(units=24, init = 'uniform', activation='relu'),
    Dropout(0.5),
    Dense(units=20, init = 'uniform', activation='relu'),
    Dense(units=24,init = 'uniform',activation='relu'),
    Dense(units=26,init = 'uniform',activation='relu'),
    Dense(units=30,init = 'uniform',activation='relu'),
    Dense(units=1,activation='sigmoid'),
])

model.summary()

#compile
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=32,epochs=100)

score = model.evaluate(X_test, y_test)

print(score)

# Initializing the ANN

ann = tf.keras.models.Sequential()
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(Dropout(0.1))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(Dropout(0.1))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(Dropout(0.1))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#more than one class (3 classes)
#ann.add(tf.keras.layers.Dense(units=3, activation='softmax'))
#compile
## Compile model
# ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

score = ann.evaluate(X_test, y_test)

print(score)


y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

f1 = f1_score(y_test, y_pred)
print(f1)

acc=accuracy_score(y_test, y_pred)
print(acc)

sns.heatmap(cm, annot=True, fmt='g')
# Use our ANN model to predict if the customer with the following informations will leave the bank:
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $ 60000
# Number of Products: 2
# Does this customer have a credit card? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $ 50000
# So, should we say goodbye to that customer?
# Solution:
# Therefore, our ANN model predicts that this customer stays in the bank!
# Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
# Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

#The bias -variance tradeoff

#tuning ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    model = Sequential([
        Dense(units=16, init='uniform', input_dim=12, activation='relu'),
        Dense(units=24, init='uniform', activation='relu'),
        Dropout(0.1),
        Dense(units=20, init='uniform', activation='relu'),
        Dense(units=24, init='uniform', activation='relu'),
        Dense(units=26, init='uniform', activation='relu'),
        Dense(units=30, init='uniform', activation='relu'),
        Dense(units=1, activation='sigmoid'),
    ])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn=build_classifier, batch_size=25, epochs=100)

accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)

# def build_classifier():
#     ann = tf.keras.models.Sequential()
#     ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#     ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#     ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#     ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#     ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#     return ann
#
# ann = KerasClassifier(build_fn=build_classifier, batch_size=32, epochs=100)
#
# accuracies = cross_val_score(estimator=ann, X=X_train, y=y_train, cv=10)

ann_mean = accuracies.mean()
print(ann_mean)
ann_variance = accuracies.std()
print(ann_variance)
print("ANN Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

#Tuning ANN
def build_classifier(optimizer, loss):
    model = Sequential([
        Dense(units=16, init='uniform', input_dim=12, activation='relu'),
        Dense(units=24, init='uniform', activation='relu'),
        Dropout(0.1),
        Dense(units=20, init='uniform', activation='relu'),
        Dense(units=24, init='uniform', activation='relu'),
        Dense(units=26, init='uniform', activation='relu'),
        Dense(units=30, init='uniform', activation='relu'),
        Dense(units=1, activation='sigmoid'),
    ])
    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [25,32],
              'nb_epoch': [100, 500, 1000],
              'optimizer': ['adam', 'rmsprop'],
              'loss' : ['binary_crossentropy','categorical_crossentropy']}

grid_search = GridSearchCV(estimator= model,
                           param_grid= parameters,
                           scoring= 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
print(best_parameters)
best_accuracy = grid_search.best_score_
print(best_accuracy)


