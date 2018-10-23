#Generative Adversarial Network.
# imports
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Reshape, UpSampling2D, BatchNormalization, Activation, Dropout, Deconvolution2D
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist

# Load data
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_train = x_train[:5000] # I took just a few for quick training
print(x_train.shape)

# Define input shape (D in random space)
in_shape = 10

#each network is created by layers, next functions concatenate the layers ...
# Generative part
g_input = Input(shape=(in_shape,))
# Block1
H = Dense(128 * 7 * 7)(g_input)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Reshape([7, 7, 128])(H)
H = Dropout(0.4)(H)
H = UpSampling2D()(H)
# Block2
H = Deconvolution2D(filters=64, kernel_size=[5, 5], padding='same')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = UpSampling2D()(H)
# Block3
H = Deconvolution2D(filters=32, kernel_size=[5, 5], padding='same')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
# Block4
H = Deconvolution2D(filters=16, kernel_size=[5, 5], padding='same')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
# Block5
H = Deconvolution2D(filters=1, kernel_size=[5, 5], padding='same')(H)
g_output = Activation('sigmoid')(H) #output of the network.
# Model
generator = Model(g_input, g_output)
generator.summary()

# Discriminative part
d_input = Input(shape=(28, 28, 1))
# B1
H = Conv2D(16, kernel_size=[5, 5], strides=2, padding='same')(d_input)
H = LeakyReLU(alpha=0.2)(H)
H = Dropout(0.4)(H)
# B2
H = Conv2D(32, kernel_size=[5, 5], strides=2, padding='same')(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dropout(0.4)(H)
# B3
H = Conv2D(64, kernel_size=[5, 5], strides=2, padding='same')(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dropout(0.4)(H)
# B4
H = Conv2D(128, kernel_size=[5, 5], strides=1, padding='same')(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dropout(0.4)(H)
# Flatten
H = Flatten()(H)
d_output = Dense(1, activation='sigmoid')(H)
# Model
discriminator = Model(d_input, d_output)
discriminator.compile(loss='binary_crossentropy', optimizer='Adam')
discriminator.summary()

# Build stacked GAN model
discriminator.trainable = False
gan_input = Input(shape=(in_shape,))
H = generator(gan_input)
gan_output = discriminator(H)
GAN = Model(gan_input, gan_output)
GAN.compile(loss='binary_crossentropy', optimizer='Adam')
GAN.summary()

# Func to plot some examples
def plot_examples(X):
    plt.figure(figsize=(20, 4))
    for i in range(len(X)):
        ax = plt.subplot(1, len(X), i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# Create initial noise
noise_train = np.random.uniform(-1, 1, size=(x_train.shape[0], in_shape))
gen_img_train = generator.predict(noise_train)
print(gen_img_train.shape)
plot_examples(gen_img_train[:10])

# Make or not trainable
#def make_trainable(model, is_train=True):
#    model.trainable = is_train
#    for l in model.layers:
#        l.trainable = is_train

# Just check the function is working
#for l in discriminator.layers:
#    print(l.trainable)
#make_trainable(discriminator, is_train=False)
#for l in discriminator.layers:
#    print(l.trainable)
#make_trainable(discriminator, is_train=True)
#for l in discriminator.layers:
#    print(l.trainable)

# Generate noisy images
#noise_train = np.random.uniform(-1, 1, size=(x_train.shape[0], in_shape))
#gen_img_train = generator.predict(noise_train)
#print(gen_img_train.shape)

# Concatenate with real images
X = np.concatenate((x_train, gen_img_train))
print(X.shape)

# Create labels (real)
Y = np.concatenate((np.ones([len(x_train), 1]), np.zeros([len(x_train), 1])))
print(Y.shape)
print(Y.sum(axis=0))

# Train discriminator (update both)
discriminator.trainable = True
discriminator.fit(X, Y, epochs=1, batch_size=128, shuffle=True)
y_hat = discriminator.predict(X)
score = discriminator.evaluate(x=X, y=Y, verbose=False)
print(f"Loss: {score:6.4f}")
print(np.round(y_hat))

# Generate noisy input signals
X = np.random.uniform(-1, 1, size=(x_train.shape[0], in_shape))
print(X.shape)

# Create labels (fake - they must fool the discriminator)
Y = np.ones([X.shape[0], 1])
print(Y.shape)
print(Y.sum(axis=0))

# Freeze discriminator and train GAN
discriminator.trainable = False
GAN.fit(X, Y, epochs=1, batch_size=128)

# Create initial noise
noise_train = np.random.uniform(-1, 1, size=(x_train.shape[0], in_shape))
gen_img_train = generator.predict(noise_train)
print(gen_img_train.shape)
plot_examples(gen_img_train[:10])

# Training
n_epochs = 50
for epoch in range(n_epochs):
    print(f"EPOCH: {epoch+1:3d} -- ", end="")

    # Generate noisy images, concatenate with real images, and assign labels
    noise_train = np.random.uniform(-1, 1, size=(x_train.shape[0], in_shape))
    gen_img_train = generator.predict(noise_train)
    X = np.concatenate((x_train, gen_img_train))
    Y = np.concatenate((np.ones([len(x_train), 1]), np.zeros([len(x_train), 1])))

    # Train discriminator (update both)
    print(f" Train discr. ", end="")
    #make_trainable(discriminator, is_train=True)
    discriminator.trainable = True
    discriminator.fit(X, Y, epochs=1, batch_size=128, shuffle=True, verbose=False)
    loss_disc = discriminator.evaluate(x=X, y=Y, verbose=False)
    print(f"Done. Loss {loss_disc:6.4f}, ", end="")

    # Generate noisy input signals and assign fake labels
    X = np.random.uniform(-1, 1, size=(x_train.shape[0], in_shape))
    Y = np.ones([X.shape[0], 1])

    # Freeze discriminator and train GAN
    print(f"-- Train GAN. ", end="")
    #make_trainable(discriminator, is_train=False)
    discriminator.trainable = False
    GAN.fit(X, Y, epochs=1, batch_size=128, verbose=False)
    loss_gen = GAN.evaluate(x=X, y=Y, verbose=False)
    print(f"Done. Loss {loss_gen:6.4f}.")

    if (epoch % 10) == 0:
        # Print examples of generated images
        noise_train = np.random.uniform(-1, 1, size=(x_train.shape[0], in_shape))
        gen_img_train = generator.predict(noise_train)
        plot_examples(gen_img_train[:10])

# Last plot
plot_examples(gen_img_train[:10])

# https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
# https://github.com/Zackory/Keras-MNIST-GAN
