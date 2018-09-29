# Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.regularizers import l2, l1

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = np_utils.to_categorical(y_train)  #
y_test = np_utils.to_categorical(y_test)


# Split train into train and validation
validation_rate = 0.2
n_train_samples = round(validation_rate * len(x_train))
print("Taking {} validation samples".format(n_train_samples))
x_val = x_train[:n_train_samples]
y_val = y_train[:n_train_samples]
x_train = x_train[n_train_samples:]
y_train = y_train[n_train_samples:]




######################MODELS #################################################
model = Sequential()
model.add(Dense(units=1000, input_dim=x_train.shape[1], activation='relu'))
#model.add(Dropout(0.2)) #adds dropout from first to second layer of 20 percent.
model.add(Dense(units=500, activation='relu',kernel_regularizer=l2(0.0001))) #it is possible to use regularizers
model.add(Dense(units=500, activation='relu',kernel_regularizer=l2(0.0001)))
model.add(Dense(units=500, activation='relu',kernel_regularizer=l2(0.0001)))
model.add(Dense(units=500, activation='relu',kernel_regularizer=l2(0.0001)))
model.add(Dense(units=10, activation='softmax'))
#for batch normalization
#model.add(Dense(units=10, kernel_regularizer=l2(00001)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))

#model.summary()
# Compiling model (define optimizer and loss function)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy') #crossentropy is a kullback-leibler divergence kind.

# Traning the model
num_epochs = 20
losses = np.zeros((num_epochs, 2))
#print(f"Training on {x_train.shape[0]} samples - validating on {x_val.shape[0]} samples.")
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1:3d} -- ", end="")
    model.fit(x_train, y_train, epochs=1, batch_size=128, validation_data=(x_val, y_val), verbose=False)
    losses[epoch, 0] = model.evaluate(x_train, y_train, verbose=False)
    losses[epoch, 1] = model.evaluate(x_val, y_val, verbose=False)
    print(f"Train loss: {losses[epoch, 0]:6.4f} -- Val loss{losses[epoch, 1]:6.4f}")

###########################Model 2#####################################
model2 = Sequential()
model2.add(Dense(units=5000, input_dim=x_train.shape[1], activation='relu'))
#model2.add(Dropout(0.2)) #adds dropout from first to second layer of 20 percent.
model2.add(Dense(units=500, activation='relu')) #,kernel_regularizer=l2(0.0001))) #it is possible to use regularizers
model2.add(Dense(units=10, activation='softmax'))
#for batch normalization
#model2.add(Dense(units=10, kernel_regularizer=l2(00001)))
#model2.add(BatchNormalization())
#model2.add(Activation('relu'))

#model2.summary()
# Compiling model2 (define optimizer and loss function)
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy') #crossentropy is a kullback-leibler divergence kind.

# Traning the model2
num_epochs = 20
losses2 = np.zeros((num_epochs, 2))
#print(f"Training on {x_train.shape[0]} samples - validating on {x_val.shape[0]} samples.")
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1:3d} -- ", end="")
    model2.fit(x_train, y_train, epochs=1, batch_size=128, validation_data=(x_val, y_val), verbose=False)
    losses2[epoch, 0] = model2.evaluate(x_train, y_train, verbose=False)
    losses2[epoch, 1] = model2.evaluate(x_val, y_val, verbose=False)
    print(f"Train loss: {losses2[epoch, 0]:6.4f} -- Val loss{losses2[epoch, 1]:6.4f}")

###########################Model 2#####################################
model3 = Sequential()
model3.add(Dense(units=5000, input_dim=x_train.shape[1], activation='relu'))
#model3.add(Dropout(0.2)) #adds dropout from first to second layer of 20 percent.
model3.add(Dense(units=500, activation='relu',kernel_regularizer=l2(0.0001))) #it is possible to use regularizers
model3.add(Dense(units=10, activation='softmax'))
#for batch normalization
#model3.add(Dense(units=10, kernel_regularizer=l2(00001)))
#model3.add(BatchNormalization())
#model3.add(Activation('relu'))

#model3.summary()
# Compiling model3 (define optimizer and loss function)
model3.compile(optimizer='rmsprop', loss='categorical_crossentropy') #crossentropy is a kullback-leibler divergence kind.

# Traning the model3
num_epochs = 20
losses3 = np.zeros((num_epochs, 2))
#print(f"Training on {x_train.shape[0]} samples - validating on {x_val.shape[0]} samples.")
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1:3d} -- ", end="")
    model3.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_val, y_val), verbose=False)
    losses3[epoch, 0] = model3.evaluate(x_train, y_train, verbose=False)
    losses3[epoch, 1] = model3.evaluate(x_val, y_val, verbose=False)
    print(f"Train loss: {losses3[epoch, 0]:6.4f} -- Val loss{losses3[epoch, 1]:6.4f}")


###########################Model 2#####################################
model4 = Sequential()
model4.add(Dense(units=1000, input_dim=x_train.shape[1], activation='relu'))
#model4.add(Dropout(0.2)) #adds dropout from first to second layer of 20 percent.
model4.add(Dense(units=500, activation='relu'))#,kernel_regularizer=l2(0.0001))) #it is possible to use regularizers
model4.add(Dense(units=500, activation='relu'))
model4.add(Dense(units=500, activation='relu'))
model4.add(Dense(units=500, activation='relu'))
model4.add(Dense(units=10, activation='softmax'))
#for batch normalization
#model4.add(Dense(units=10, kernel_regularizer=l2(00001)))
#model4.add(BatchNormalization())
#model4.add(Activation('relu'))

#model4.summary()
# Compiling model4 (define optimizer and loss function)
model4.compile(optimizer='rmsprop', loss='categorical_crossentropy') #crossentropy is a kullback-leibler divergence kind.

# Traning the model4
num_epochs = 20
losses4 = np.zeros((num_epochs, 2))
#print(f"Training on {x_train.shape[0]} samples - validating on {x_val.shape[0]} samples.")
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1:3d} -- ", end="")
    model4.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_val, y_val), verbose=False)
    losses4[epoch, 0] = model4.evaluate(x_train, y_train, verbose=False)
    losses4[epoch, 1] = model4.evaluate(x_val, y_val, verbose=False)
    print(f"Train loss: {losses4[epoch, 0]:6.4f} -- Val loss{losses4[epoch, 1]:6.4f}")


######################PLOTS######################################
# Plot training history
plt.figure(figsize=(15, 15))
ax = plt.subplot(2, 2, 1)
ax.plot(losses[:, 0], label='Training', linewidth=2)
ax.plot(losses[:, 1], label='Validation', linewidth=2)
ax.set_title("Model 1", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.xlabel("Epoch", fontsize=18)
ax.tick_params(labelsize=18)
ax = plt.subplot(2, 2, 2)
ax.plot(losses2[:, 0], label='Training', linewidth=2)
ax.plot(losses2[:, 1], label='Validation', linewidth=2)
ax.set_title("Model 2", fontsize=18)
ax.tick_params(labelsize=18)
plt.xlabel("Epoch", fontsize=18)
ax = plt.subplot(2, 2, 3)
ax.plot(losses3[:, 0], label='Training', linewidth=2)
ax.plot(losses3[:, 1], label='Validation', linewidth=2)
ax.set_title("Model 3", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.xlabel("Epoch", fontsize=18)
ax.tick_params(labelsize=18)
ax = plt.subplot(2, 2, 4)
ax.plot(losses4[:, 0], label='Training', linewidth=2)
ax.plot(losses4[:, 1], label='Validation', linewidth=2)
ax.set_title("Model 4", fontsize=18)
ax.tick_params(labelsize=18)
plt.xlabel("Epoch", fontsize=18)
#plt.legend(fontsize=18)



plt.legend(fontsize=18)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.ylim([0, 0.5])
plt.tick_params(labelsize=18)


###########################PREDICTIONS ########################################
# Make predictions for test set and evaluate performance
y_hat = model.predict(x_test)
test_loss = model.evaluate(x_test, y_test)
print("Test error: {:6.4f}".format(test_loss))

for i in range(10):
    print(y_test[i])
    print(np.round(y_hat[i]))
    print("\n")

######################This is the confusion mamtrix########################
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_hat, axis=1)
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.imshow(conf_matrix, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
