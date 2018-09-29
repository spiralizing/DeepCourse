import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import keras
# esto es para plotear en linea
from sklearn import datasets
# se importan los datos
iris = datasets.load_iris()
#hiper-parametros (epocas y velocidad de aprendizaje)
n_epochs = 10000
learn_rate = 0.01
#Variables, caso de tabla de verdad AND
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])
#el siguiente es para los datos de iris
X = iris.data[:100:2, :2]
y = iris.target[:100:2]
yp = iris.target[1:100:2]
#Inicializar parametros

sz = np.size(X[1]) +1 #numero de variables + 1
W = np.random.rand(sz)

#activacion
def step_activation(z):
    return 1 if z >=0 else 0

def sig_activation(z):
    #funcion sigmoide
    return 1/(1 + np.exp(-z))


#prediccion
def predict(W,x):
    z = W.T.dot(x)
    a = sig_activation(z)
    return a


##################entrenamiento#################################################
epoch_error = list()
for epoch in range(n_epochs):
    err = list()
    for i in range(len(X)):
        x = np.insert(X[i], 0, 1)
        y_hat = predict(W,x)
        e = y[i] - y_hat
        df = y_hat*(1-y_hat) #derivada de la funcion de activacion, en este caso simoide
        err.append(e)
        W = W + learn_rate * e * x * df
    #print(f"epoch {epoch} -- mean error: {np.array(err).mean()}")
    epoch_error.append(np.array(err).mean())

#grafica del error promedio
plt.plot(epoch_error)

P = iris.data[1:100:2, :2]
predictions = list()

for i in range(np.shape(P)[0]):
    p = np.insert(P[i],0,1)
    predictions.append(predict(W,p))

plt.plot(predictions)


plt.scatter(X.T[0],X.T[1], alpha=0.2)
