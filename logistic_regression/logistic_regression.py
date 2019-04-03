import pandas as pd
import numpy as np


class LogisticRegression():
    """
        Logistic Regression

        Parameters
        ----------
        max_number_of_iterations : int, Optional (default = 2000)
        Maximum number of iterations for training.

        learning_rate : int, Optional (default  = 0.01)
        Rate at which the algorithm learns.

        verbose : boolean, Optional (default=False)
        Controls verbosity of output:
        - False: No Output
        - True: Displays the cost at every 100th iteration.

        Attributes
        ----------
        costs_ : array, shape=[max_number_of_iterations/20]
        Returns an array with the costs.

        weights_ : array, shape=[n_samples, 1]
        Returns the weights.

        bias_: float
        Returns the bias value.
        
    """

    def __init__(self, max_number_of_iterations=2000, learning_rate=0.01, verbose=False):
        self.max_number_of_iterations = max_number_of_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose

    def fit(self, X, Y):
	"""
        Fits logistic regression to the data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        Y : array-like, shape = [n_samples]
            The target values.

		Returns
        -------
        self : object

	"""
        return self._fit(X,Y)

    def predict(self, X, Y):
	"""
        Predicts the outcome and returns the accuracy of the given (data, label)
        pair.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        Y : array-like, shape = [n_samples]
            The target values.

		Returns
        -------
        accuracy: float
            Accuracy of the given data.
	"""
        return self._predict(X, Y)

    def _fit(self, X, Y):
        return self._model(X, Y)

    def _initialize_parameters(self, dim):
        """
        Initialize the weights and the bias (Initialized as zeros).
        """
        w = np.zeros(shape = (dim, 1), dtype = np.float32)
        b = 0
        # Check the shape of the parameters.
        assert(w.shape == (dim,1))
        assert(isinstance(b, float) or isinstance(b, int))
        return w, b

    def _sigmoid(self, Z):
        """
        Perform the sigmoid activation function
        """
        return 1 / (1 + np.exp(-Z))

    def _propagation(self, w, b, X, Y):
        """
        Calculates the overall cost and gradients.
        """
        m = X.shape[1]
        A = self._sigmoid(np.dot(w.T, X) + b)
        # Calculate the overall cost.
        cost = (-1. / m) * np.sum((Y * np.log(A) + (1-Y) * np.log(1-A)), axis=1)

        # Calculate the gradient values.
        dw = (1. / m) * np.dot(X,((A-Y).T))
        db = (1. / m) * np.sum((A-Y), axis=1)

        # Check the shape of the gradients.
        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        gradients = {"dw":dw, "db":db}
        return gradients, cost

    def _optimize(self, w, b, X, Y):
        """
        Optimize the parameters(Weights and bias) using the gradients.
        """
        costs = []
        for i in range(self.max_number_of_iterations):
            gradients, cost = self._propagation(w=w, b=b, X=X, Y=Y)
            dw = gradients["dw"]
            db = gradients["db"]
            # Improve the weights and bias.
            w = w - self.learning_rate*dw
            b = b - self.learning_rate*db

            if(i%20 == 0):
                # Store the cost at every 20th iteration.
                costs.append(cost)
            if(self.verbose and i%100 == 0):
                # Display the cost at every 100th iteration.
                print "The cost after iteration %d is %f" % (i,cost)

        parameters = {"w":w, "b":b}
        gradients = {"dw":dw, "db":db}
        self.costs_ = costs
        return parameters,gradients,costs

    def _model(self, X_train, Y_train):
        """
        Runs the entire algorithm.
        """
        w,b = self._initialize_parameters(X_train.shape[0])
        parameters, gradients, costs = self._optimize(w, b, X_train, Y_train)
        self.weights_ = parameters["w"]
        self.bias_ = parameters["b"]

    def _predict(self, X, Y):
        """
        Gets the Prediction for given data and returns the accuracy.
        """
        # Checks if fit(X,Y) has been called before predict(X, Y) is called.
        try:
			self.weights_
        except AttributeError:
			raise ValueError('fit(X, Y) needs to be called before using predict(X,Y).')
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        self.weights_ = self.weights_.reshape(X.shape[0], 1)
        A = self._sigmoid(np.dot(self.weights_.T, X) + self.bias_)
        # Get the prediction of the given data.
        for i in range(A.shape[1]):
            Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0
        assert(Y_prediction.shape == (1,m))
        accuracy = (100 - np.mean(np.abs(Y_prediction - Y))*100)
        return accuracy
