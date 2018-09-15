from sklearn import datasets
import matplotlib.pyplot as plt
import DL_Classes as DL #se importa la clase "perceptron"
# Import the Iris dataset
iris = datasets.load_iris()
X = iris.data[:100, :]  # Features: Take just the first 2 dimensions from the first 100 elements.
y = iris.target[:100]

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()


# Train one perceptron to classify data
percp = DL.Perceptron(eta=0.1, num_epochs=50)
percp.fit(X, y)


plt.plot(range(1, len(percp.errors_) + 1), percp.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Classification error')
plt.show()
