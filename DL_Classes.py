import numpy as np
class Perceptron(object):
    """ Perceptron classifier.

    Parameters
    ----------
    eta: float
        Learning rate (between 0.0 and 1.0).
    num_epochs: int
        Number of epochs.

    Attributes
    ----------
    w_: 1d-array
        Weights after fitting.
    errors_: list
        Number of missclassified instances per epoch.
    """

    def __init__(self, eta=0.01, num_epochs=10):
        self.eta = eta
        self.num_epochs = num_epochs

    def fit(self, X, y):
        """ Fit training data

        Parameters
        -----------
        X: {array-like}, shape = [num_samples, num_features]
            Training vectors.
        y: array-like, shape = [n_samples]
            Target values.

        Returns
        -----------
        self: object
        """
        self.w_ = np.random.rand(1 + X.shape[1]) - 0.5
        self.errors_ = []

        for _ in range(self.num_epochs):
            errors = 0
            for x, target in zip(X, y):
                err = target - self.predict(x)
                update = self.eta * err
                self.w_[1:] += update * x
                self.w_[0] += update
                errors += err
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """ Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
