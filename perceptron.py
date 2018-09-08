import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# esto es para plotear en linea

#hiper-parametros (epocas y velocidad de aprendizaje)
n_epochs = 100
learn_rate = 0.5
#Variables, caso de tabla de verdad AND
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

#Inicializar parametros
W = np.random.rand(3)

#activacion
def activation(z):
    return 1 if z >=0 else 0

#prediccion
def predict(W,x):
    z = W.T.dot(x)
    a = activation(z)
    return a

np.insert
#entrenamiento
epoch_error = list()
for epoch in range(n_epochs):
    err = list()
    for i in range(len(X)):
        x = np.insert(X[i], 0, 1)
        y_hat = predict(W,x)
        e = y[i] - y_hat
        err.append(e)
        W = W + learn_rate * e * x
    #print("epoch {} -- mean error: {}".format(epoch, np.array(err).mean()))
    epoch_error.append(np.array(err).mean())

#grafica del error promedio
plt.plot(epoch_error)
