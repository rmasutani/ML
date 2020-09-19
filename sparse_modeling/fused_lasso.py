import numpy as np 

class FusedLasso(object):
    
    def __init__(self, lambda_=0.1, max_iter=1000, tol=0.0001):
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self._weights = None
        self.tol = tol
    
    def fit(self, X):
        img_shape = X.shape
        X = X.ravel()
        self._weights = np.random.normal(size=(X.shape))
        M = self._create_adj_matrix(img_shape)
        
        avgl1 = 0.
        
        for _ in range(self.max_iter):
            avgl1_prev = avgl1
            self._weights -= self._calc_grad(X, X.shape[0], M)
            avgl1 = np.abs(self._weights).sum() / self._weights.shape[0]
            if abs(avgl1 - avgl1_prev) <= self.tol:
                print("break at iter={}".format(_))
                break
    
    def fit_transform(self, X):
        self.fit(X)
        return self._weights.reshape(X.shape)
    
    def _soft_threshold(self):
        return 
            
    def _create_adj_matrix(self, shape):
        H, W = shape
        M = np.zeros((H*W, 4), int)
        
        for h in range(H):
            for w in range(W):
                idx = h*H + w
                M[idx, 0] = (h-1)*H + w if (h-1) >= 0 else idx # above
                M[idx, 1] = h*H + (w-1) if (w-1) >= 0 else idx # left
                M[idx, 2] = h*W + (w+1) if (w+1) < W else idx # right
                M[idx, 3] = (h+1)*H + w if (h+1) < H else idx # bottom
        
        return M
    
    def _calc_grad(self, X, n, adj):
        # Calculate subgrads
        subgrad = np.zeros_like(self._weights)
        for i in range(adj.shape[1]):
            subgrad += np.sign(self._weights - self._weights[adj[:, i]])
        
        grad = (0.01) * (self._weights - X) + self.lambda_ * subgrad
        
        return grad
        