# imports
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras import regularizers
from keras.datasets import mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)

# Model definition
hidden_sizes = [32, 2]
model = Sequential()
model.add(Dense(units=hidden_sizes[0], input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dense(units=hidden_sizes[1], activation='relu'))
model.add(Dense(units=hidden_sizes[0], activation='relu'))
model.add(Dense(units=x_train.shape[1], activation='sigmoid'))
model.summary()

# Compile
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.fit(x_train, x_train, epochs=10, batch_size=128, shuffle=True, validation_split=0.2)

# Create a new model (encoder) taking parts of previous model
encoder = Model(inputs=model.input, outputs=model.get_layer(index=1).output)

# Apply it on new data
x_encoded = encoder.predict(x_test)
print(x_encoded.shape)

# Plot embeddings
plt.figure(figsize=(12, 10))
for cl in range(10):
    plt.scatter(x_encoded[y_test==cl, 0], x_encoded[y_test==cl, 1], s=1, label=cl)
plt.legend()
plt.grid()

# Reconstruction encoding-decoding
x_decoded = model.predict(x_test)
print(x_decoded.shape)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_decoded[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
