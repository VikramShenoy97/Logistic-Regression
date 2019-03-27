import pandas as pd
import numpy as np


class LogisticRegression():

    def __init__(self, max_number_of_iterations=2000, learning_rate=0.01, verbose=False):
        self.max_number_of_iterations = max_number_of_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose

    def fit(self, X, Y):
        return self._fit(X,Y)

    def predict(self, X, Y):
        return self._predict(X, Y)

    def evaluate_loss(self):
        return self._evaluate_loss()

    def _fit(self, X, Y):
        return self._model(X, Y)

    def _initialize_parameters(self, dim):
        w = np.zeros(shape = (dim, 1), dtype = np.float32)
        b = 0
        assert(w.shape == (dim,1))
        assert(isinstance(b, float) or isinstance(b, int))
        return w, b

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _propagation(self, w, b, X, Y):
        m = X.shape[1]
        A = self._sigmoid(np.dot(w.T, X) + b)
        cost = (-1. / m) * np.sum((Y * np.log(A) + (1-Y) * np.log(1-A)), axis=1)

        dw = (1. / m) * np.dot(X,((A-Y).T))
        db = (1. / m) * np.sum((A-Y), axis=1)

        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        gradients = {"dw":dw, "db":db}
        return gradients, cost

    def _optimize(self, w, b, X, Y):
        costs = []
        for i in range(self.max_number_of_iterations):
            gradients, cost = self._propagation(w=w, b=b, X=X, Y=Y)
            dw = gradients["dw"]
            db = gradients["db"]
            w = w - self.learning_rate*dw
            b = b - self.learning_rate*db

            if(i%20 == 0):
                costs.append(cost)
            if(self.verbose and i%100 == 0):
                print "The cost after iteration %d is %f" % (i,cost)

        parameters = {"w":w, "b":b}
        gradients = {"dw":dw, "db":db}
        self.costs_ = costs
        return parameters,gradients,costs

    def _model(self, X_train, Y_train):
        w,b = self._initialize_parameters(X_train.shape[0])
        parameters, gradients, costs = self._optimize(w, b, X_train, Y_train)
        self.weights_ = parameters["w"]
        self.bias_ = parameters["b"]

    def _predict(self, X, Y):
        try:
			self.weights_
        except AttributeError:
			raise ValueError('fit(X, Y) needs to be called before using predict(X,Y).')
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        self.weights_ = self.weights_.reshape(X.shape[0], 1)
        A = self._sigmoid(np.dot(self.weights_.T, X) + self.bias_)
        for i in range(A.shape[1]):
            Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0
        assert(Y_prediction.shape == (1,m))
        accuracy = (100 - np.mean(np.abs(Y_prediction - Y))*100)
        return accuracy

    def _evaluate_loss(self):
        try:
			self.weights_
        except AttributeError:
			raise ValueError('fit(X, Y) needs to be called before using evaluate_loss().')
        return self.costs_
