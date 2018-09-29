# Imports
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
model.add(Dense(units=100, input_dim=x_train.shape[1], activation='relu'))
#model.add(Dropout(0.2)) #adds dropout from first to second layer of 20 percent.
model.add(Dense(units=50, activation='relu',kernel_regularizer=l2(0.0001))) #it is possible to use regularizers
model.add(Dense(units=10, activation='softmax'))
#for batch normalization
#model.add(Dense(units=10, kernel_regularizer=l2(00001)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))

#model.summary()
# Compiling model (define optimizer and loss function)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy') #crossentropy is a kullback-leibler divergence kind.

# Train your model
num_epochs = 20
losses = np.zeros((num_epochs, 2))
#print(f"Training on {x_train.shape[0]} samples - validating on {x_val.shape[0]} samples.")
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1:3d} -- ", end="")
    model.fit(x_train, y_train, epochs=1, batch_size=128, validation_data=(x_val, y_val), verbose=False)
    losses[epoch, 0] = model.evaluate(x_train, y_train, verbose=False)
    losses[epoch, 1] = model.evaluate(x_val, y_val, verbose=False)
    print(f"Train loss: {losses[epoch, 0]:6.4f} -- Val loss{losses[epoch, 1]:6.4f}")



######################PLOTS######################################
# Plot training history
plt.figure(figsize=(15, 10))
plt.plot(losses[:, 0], label='Training', linewidth=2)
plt.plot(losses[:, 1], label='Validation', linewidth=2)
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
