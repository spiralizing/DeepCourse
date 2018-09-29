import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
#############importing keras##########
from keras import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.datasets import mnist
#####################################################
#se asignan los datos a las variables.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # aqui se reacomodan los datos para su mejor lectura...
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
np.prod
x_train

#las siguientes lineas son para imprimir las imagenes...

plt.figure(figsize=(20, 4))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# Print frequencies of each class, i.e., number of instances per class
plt.figure(figsize=(12, 4))
plt.hist(y_train, np.linspace(-0.5, 9.5, 11), rwidth=0.9)
plt.title("Class distribution on the training set")
plt.xlabel("Class label (digit)")
plt.ylabel("Class frequency")
plt.grid()

# Extract only two classes, e.g., classes 0 and 1
class0 = 0 # Try with other class by modifying this parameter
class1 = 1 # Try with other class by modifying this parameter


# Training set
training_indices = np.logical_or(y_train == class0, y_train == class1) # identify indices of selected classes, (is a filter, returns false or true)
X_training = x_train[training_indices] # These are the filtered matrices of values (features)
Y_training = y_train[training_indices] # These are the filtered classes

#Y_training
Y_training[Y_training == class0] = 0 # Assign value of 0 to one class
Y_training[Y_training == class1] = 1 # and value of 1 to the other one.
print("Size of training set")
print(X_training.shape)
print(Y_training.shape)

# Test set
training_indices = np.logical_or(y_test == class0, y_test == class1) # identify indices of selected classes
X_testing = x_test[training_indices] # Copy features of identified elements
Y_testing = y_test[training_indices] # Copy labels of identified elements
Y_testing[Y_testing == class0] = 0 # Assign value of 0 to one class
Y_testing[Y_testing == class1] = 1 # and value of 1 to the other one.
print("Size of test set")
print(X_testing.shape)
print(Y_testing.shape)

###################################### Let's try 3 different models#############
###################################### Initializing the model 1 #######

model= Sequential()
model.add(Dense(units=10, input_dim=X_training.shape[1], activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
#model.summary()

# Compile model
sgd = optimizers.SGD(lr=0.01) # se define el learning rate
model.compile(optimizer='sgd', loss='mean_squared_error')

#training
hist = model.fit(X_training, Y_training, epochs=10, batch_size=100)

###################################### Initializing the model #######

model2= Sequential()
model2.add(Dense(units=100, input_dim=X_training.shape[1], activation='relu'))
model2.add(Dense(units=10, activation='relu'))
model2.add(Dense(units=1, activation='sigmoid'))
#model2.summary()

# Compile model
sgd = optimizers.SGD(lr=0.01) # se define el learning rate
model2.compile(optimizer='sgd', loss='mean_squared_error')

#training
hist2 = model2.fit(X_training, Y_training, epochs=10, batch_size=100)

###################################### Initializing the model #######

model3= Sequential()
model3.add(Dense(units=20, input_dim=X_training.shape[1], activation='relu'))
model3.add(Dense(units=8, activation='relu'))
model3.add(Dense(units=1,activation='sigmoid'))
#model3.summary()

# Compile model3
sgd = optimizers.SGD(lr=0.01) # se define el learning rate
model3.compile(optimizer='sgd', loss='mean_squared_error')

#training
hist3 = model3.fit(X_training, Y_training, epochs=10, batch_size=100)

################################Plotting #######################################

# Plot training history
plt.figure(figsize=(10, 10))
plt.plot(hist.history['loss'], label='model1')
plt.plot(hist2.history['loss'], label='model2')
plt.plot(hist3.history['loss'], label='model3')
plt.legend("123")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Make predictions for test set
y_hat = model.predict(X_testing)
y_hat2 = model2.predict(X_testing)
y_hat3 = model3.predict(X_testing)
# Plot info about classes: test set and prediction
plt.figure(figsize=(18, 10))
ax = plt.subplot(2, 2, 1)
ax.plot(Y_testing, '.')
ax.set_title("Test Dataset")
ax = plt.subplot(2, 2, 2)
ax.plot(y_hat, '.')
ax.set_title("Prediction model 1")
ax = plt.subplot(2, 2, 3)
ax.plot(y_hat2, '.')
ax.set_title("Prediction model 2")
ax = plt.subplot(2, 2, 4)
ax.plot(y_hat3, '.')
ax.set_title("Prediction model 3")

# Evaluate the prediction error on the test set (data not seen during training. Good to see how well our will generalize)
test_error = model.evaluate(X_testing, Y_testing)
test_error2 = model2.evaluate(X_testing, Y_testing)
test_error3 = model3.evaluate(X_testing, Y_testing)


test_error
test_error2
test_error3
