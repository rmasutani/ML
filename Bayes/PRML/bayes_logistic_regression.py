import numpy as np 

class LogisticRegression():
    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

class BayesianLogisticRegression(LogisticRegression):
    """
    Logistic regression model
    w ~ Gaussian(0, alpha^(-1)I)
    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """

    def __init__(self, alpha:float=1.):
        self.alpha = alpha

    def fit(self, X:np.ndarray, t:np.ndarray, max_iter:int=100):
        """
        bayesian estimation of logistic regression model
        using Laplace approximation
        Parameters
        ----------
        X : (N, D) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
            binary 0 or 1
        max_iter : int, optional
            maximum number of paramter update iteration (the default is 100)
        """
        w = np.zeros(np.size(X, 1))
        eye = np.eye(np.size(X, 1))
        self.w_mean = np.copy(w)
        self.w_precision = self.alpha * eye
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self._sigmoid(X @ w)
            grad = X.T @ (y - t) + self.w_precision @ (w - self.w_mean)
            hessian = (X.T * y * (1 - y)) @ X + self.w_precision
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w_mean = w
        self.w_precision = hessian
    
    def predict_proba(self, X):
        return self._sigmoid(X @ self.w_mean)

    def predict_dist(self, X:np.ndarray):
        """
        compute probability of input belonging class 1
        Parameters
        ----------
        X : (N, D) np.ndarray
            training data independent variable
        Returns
        -------
        (N,) np.ndarray
            probability of positive
        """
        mu_a = X @ self.w_mean
        var_a = np.sum(np.linalg.solve(self.w_precision, X.T).T * X, axis=1)

        print(var_a.shape, mu_a.shape)

        print(var_a)

        return self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))

