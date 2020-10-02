import numpy as np 

class GaussianProcessRegression(object):
    """
    Gaussian process regression

    - Model 
        y = f(x) + eps

        y: target variable
        x: input variable
        eps: noise
        
    - Distributions    
        eps_n ~ N(0, sigma^2)
        p(y|f) ~ N(f, sigma^2*I)
    
    - Kernel (default "gaussian")
        K: Gram matrix
    """

    def __init__(self, kernel_type="gaussian", theta1=1, theta2=1, var_eps=0.1):
        self.kernel_type = kernel_type
        self.theta1 = theta1
        self.theta2 = theta2
        self.var_eps = var_eps
        self.gram_matrix = None
        self.gram_matrix_pred = None

    def _kernel(self, x1, x2):
        if self.kernel_type == "gaussian":
            out = self.theta1 * np.exp(-np.linalg.norm(x1-x2)**2 / self.theta2)
        else:
            # Linear kernel
            out = np.inner(x1, x2)

        return out

    def _make_gram_matrix(self, x):
        size = len(x)
        K = np.zeros(shape=(size, size))

        for i in range(size):
            for j in range(size):
                K[i, j] = self._kernel(x[i], x[j])

        return K 

    def fit(self, X):
        size = len(X)
        self.gram_matrix = self._make_gram_matrix(X) + self.var_eps * np.eye(size)

    def predict_dist(self, x_train, x_test, y):
        # Calculate new gram matrix
        n = self.gram_matrix.shape[0] # size of the exsiting gram matrix
        m = len(x_test) # number of points (new ones included)
        k1 = np.zeros(shape=(n, m))
        for i in range(n):
            for j in range(m):
                k1[i, j] = self._kernel(x_train[i], x_test[j])
        
        k2 = self._make_gram_matrix(x_test)

        # stack
        # tmp1 = np.hstack((self.gram_matrix, k1))
        # tmp2 = np.hstack((k1.T, k2))
        # self.gram_matrix_pred = np.vstack((tmp1, tmp2)) 
        K_inv = np.linalg.inv(self.gram_matrix)
        mean = k1.T @ K_inv @ y
        cov = k2 - k1.T @ K_inv @ k1       
        
        return mean, cov

    # def plot_sample(self):
    #     pass   
