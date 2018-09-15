import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
#############importing keras##########
from keras import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.datasets import mnist
#####################################################
#datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()
