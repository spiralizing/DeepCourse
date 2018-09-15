#This file is a construction of multi-later perceptron
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
#############importing keras##########
from keras import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

#################Start################
# Import the Iris dataset
iris = datasets.load_iris()
X = iris.data[:100, :]  # Features: Take just the first 2 dimensions from the first 100 elements.
y = iris.target[:100]

#Build model
model = Sequential() #secuencial es que agrega los nodos hacia la derecha
model.add(Dense(units=1, input_dim=X.shape[1])) #Se agregan unidades (neuronas) con el numero de inputs
model.add(Activation('sigmoid')) #tipo de activacion
#model.summary()

# Compile model
sgd = optimizers.SGD(lr=0.01) # se define el learning rate
model.compile(optimizer='sgd', loss='mean_squared_error')

#fitting
model.fit(X, y, epochs=50, batch_size=5)
######################################


###################################### Different layers #######

model2= Sequential()
model2.add(Dense(units=2, input_dim=X.shape[1], activation='relu'))
model2.add(Dense(units=5, activation='relu'))
model2.add(Dense(units=1, activation='sigmoid'))
model2.summary()

# Compile model
sgd = optimizers.SGD(lr=0.01) # se define el learning rate
model2.compile(optimizer='sgd', loss='mean_squared_error')

#fitting
hist = model2.fit(X, y, epochs=50, batch_size=5)

y_pred = model2.predict(X)


plt.plot(y_pred) #ver las predicciones
plt.plot(y)
plt.show()

yp_r = np.round(y_pred)
plt.plot(yp_r)

plt.plot(hist.history['loss']) #ver la evolucion de los errores promedio.
