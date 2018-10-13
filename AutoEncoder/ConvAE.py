# imports
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Reshape, UpSampling2D
from keras.models import Sequential, Model
from keras.regularizers import l1
from keras.datasets import mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
print(x_train.shape)

# Add Gaussian noise
noise_factor = 0.25
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, a_min=0.0, a_max=1.0)
x_test_noisy = np.clip(x_test_noisy, a_min=0.0, a_max=1.0)

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
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


# Model definition
autoencoder = Sequential()
autoencoder.add(Conv2D(input_shape=(28, 28, 1), filters=16, kernel_size=(5, 5), activation='relu', padding='same'))
autoencoder.add(MaxPool2D(pool_size=(2, 2)))
autoencoder.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPool2D(pool_size=(2, 2)))
# Here the output is (32 layers of 7x7 = 1568D)

# Flatten and compute hidden (short) sparse representation
autoencoder.add(Flatten())
autoencoder.add(Dense(units=32, activity_regularizer=l1(3e-4), activation='relu'))

# Go back to layers format
autoencoder.add(Dense(units=1568, activation='relu'))
autoencoder.add(Reshape(target_shape=(7, 7, 32)))
# Here the output is the same shape and size as last MaxPool or Conv

# Create decoding layers
autoencoder.add(UpSampling2D(size=(2, 2)))
autoencoder.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2, 2)))
autoencoder.add(Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same'))
autoencoder.summary()

# Compile
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train, epochs=10, batch_size=256, shuffle=True, validation_split=0.2)

# Reconstruction encoding-decoding
x_decoded = autoencoder.predict(x_test_noisy)
print(x_decoded.shape)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_decoded[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
