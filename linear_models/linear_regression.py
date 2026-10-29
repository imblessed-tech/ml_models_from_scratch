import numpy as np

class LinearRegressionFromScratch:
    """
    A unified, vectorized linear regression model supporting Batch, Stochastic, 
    and Mini-Batch Gradient Descent along with Ridge, Lasso, and Elastic Net regularization.
    """

    def __init__(self, method='batch', penalty=None, learning_rate=0.01, lmbda=1.0, r=0.5, batch_size=32, n_epochs=50, n_iterations=1000):
        # Hyperparameter Configuration
        self.method = method
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.r = r
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        
        # Model States & Learned Weights
        self.bias = None
        self.weights = None
        self.W = None
        self.loss_history = []
        
        # Data Scaling Memory States
        self.X_mean = None
        self.X_std = None

    def _check_is_fitted(self):
        """Validates if the model parameters have been trained before inference."""
        if self.W is None:
            raise RuntimeError(
                "This LinearRegressionFromScratch instance is not fitted yet. "
                "Call 'fit' with appropriate training data before using this method."
            )


    def _standardize_features(self, X, fit=False):
        """Computes or applies training data Z-scores to ensure stable gradients."""
        if fit:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)            
        return (X - self.X_mean) / (self.X_std + 1e-15)

    def _compute_penalty(self, W, current_m):
        """Calculates cost penalty values and structural gradient updates for regularization."""
        W_no_bias = np.copy(W)
        W_no_bias[0] = 0.0

        if self.penalty == "ridge":
            l2_penalty = (self.lmbda / current_m) * np.sum(W_no_bias **2)
            l2_grad = (2 * self.lmbda / current_m) * W_no_bias
            return l2_penalty, l2_grad
        elif self.penalty == "lasso":
            l1_penalty = (self.lmbda / current_m) * np.sum(np.abs(W_no_bias))
            l1_grad = (self.lmbda / current_m) * np.sign(W_no_bias)
            return l1_penalty, l1_grad
        elif self.penalty == "elastic_net":
            l1 = (self.lmbda * self.r / current_m) * np.sum(np.abs(W_no_bias))
            l2 = (self.lmbda * (1 - self.r) / (2 * current_m)) * np.sum(W_no_bias ** 2)
            en_penalty = l1 + l2

            l1_grad = (self.lmbda * self.r / current_m) * np.sign(W_no_bias)
            l2_grad = (2 * self.lmbda * (1 - self.r) / current_m) * W_no_bias
            en_grad = l1_grad + l2_grad
            return en_penalty, en_grad
        elif self.penalty is None:
            return 0.0, np.zeros_like(W)
        else:
            raise ValueError("Penalty type not recognized")

    def fit(self, X, y):
        """
        Learns the optimal weights and bias matrices from the training dataset.
        Handles the design matrix stack and executes the configured optimization loops.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self._standardize_features(X, fit=True)

        n_samples, n_features = X_scaled.shape
        X_new = np.column_stack([np.ones(n_samples), X_scaled])

        if self.method == "closed_form":
            coefficients, residuals, _, _ = (
                np.linalg.lstsq(X_new, y, rcond=None)
            )

            self.weights = coefficients[1:]
            self.bias = coefficients[0]
            self.W = coefficients
            return self

        else:
            W = np.zeros(n_features + 1)
            loss_history = [] 

            if self.method == "batch":
                for _ in range(self.n_iterations):
                    y_pred = np.dot(X_new, W)
                    error = y_pred - y

                    r_penalty, r_grad = self._compute_penalty(W, current_m=n_samples)

                    mse = np.mean(error ** 2) + r_penalty
                    loss_history.append(mse)

                    dw = (2 / n_samples) * np.dot(X_new.T, error) + r_grad
                    W -= self.learning_rate * dw
                self.W = W
                self.weights = W[1:]
                self.bias = W[0]
                self.loss_history = loss_history

            elif self.method == "stochastic":
                for epoch in range(self.n_epochs):
                        indices = np.random.permutation(n_samples)

                        for i in indices:
                            Xi = X_new[i]
                            yi = y[i]

                            yi_pred = np.dot(Xi, W)
                            error = yi_pred - yi
                            _, r_grad = self._compute_penalty(W, current_m=1)

                            dw = 2 * error * Xi + r_grad
                            W -= self.learning_rate * dw

                        r_penalty, _ = self._compute_penalty(W, current_m=n_samples)
                        epoch_y_pred = np.dot(X_new, W)
                        epoch_error = epoch_y_pred - y
                        epoch_loss = np.mean(epoch_error ** 2) + r_penalty

                        loss_history.append(epoch_loss)

                self.W = W
                self.weights = W[1:]
                self.bias = W[0]
                self.loss_history = loss_history


            elif self.method == "mini_batch":
                for epoch in range(self.n_epochs):
                    indices = np.random.permutation(n_samples)
                    X_shuffled = X_new[indices]
                    y_shuffled = y[indices]

                    for i in range(0, n_samples, self.batch_size):
                        X_batch = X_shuffled[i : i + self.batch_size]
                        y_batch = y_shuffled[i : i + self.batch_size]

                        k = X_batch.shape[0]

                        y_batch_pred = np.dot(X_batch, W)
                        batch_error = y_batch_pred - y_batch

                        _, r_grad = self._compute_penalty(W, current_m=k)
                        dw = (2.0 / k) * np.dot(X_batch.T, batch_error) + r_grad
                        W -= self.learning_rate * dw

                    r_penalty, _ = self._compute_penalty(W, current_m=n_samples)
                    epoch_y_pred = np.dot(X_new, W)
                    epoch_error = epoch_y_pred - y
                    epoch_loss = np.mean(epoch_error ** 2) + r_penalty
                    loss_history.append(epoch_loss)

                self.W = W
                self.weights = W[1:]
                self.bias = W[0]
                self.loss_history = loss_history

    def predict(self, X):
        """
        Generates predictions for new instances by applying the learned parameters.
        Automatically scales raw feature columns using the saved training memory states.
        """
        self._check_is_fitted()
        X_scaled = self._standardize_features(X, fit=False)

        n_samples = X_scaled.shape[0]
        X_new = np.column_stack([np.ones(n_samples), X_scaled])
        y_pred = np.dot(X_new, self.W)

        return y_pred
    
    def score(self, X, y):
        """Calculates and returns the clean R² percentage match for the provided dataset."""
        y_pred = self.predict(X)
        y_true = np.asarray(y, dtype=float).reshape(-1)
        y_mean = np.mean(y_true)
        total_variation = np.sum((y_true - y_mean) **2)
        residual_variation = np.sum((y_true - y_pred) ** 2)

        r2 = 1.0 - (residual_variation / total_variation  + 1e-15)
        
        return r2
    

class PolynomialRegressionFromScratch(LinearRegressionFromScratch):
    """
    A polynomial regression estimator that extends the base linear regression class.
    Automatically engineers higher-degree feature transformations during fit and inference.
    """
    def __init__(self, 
                degree=2, 
                method='batch', 
                penalty=None, 
                learning_rate=0.01, 
                lmbda=1.0, 
                r=0.5, 
                batch_size=32, 
                n_epochs=50, 
                n_iterations=1000
                ):
        
        super().__init__(
            method=method, 
            penalty=penalty, 
            learning_rate=learning_rate, 
            lmbda=lmbda, 
            r=r, 
            batch_size=batch_size, 
            n_epochs=n_epochs, 
            n_iterations=n_iterations
        )
        self.degree = degree

    def _check_is_fitted(self):
        """Validates if the model parameters have been trained before inference."""
        super()._check_is_fitted()
        return self

    def _generate_polynomial_features(self, X):
        """
        Transforms a raw feature matrix X into an expanded polynomial space.
        Example: For degree=3, converts [X] into [X, X^2, X^3]
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        poly_columns = []
        for d in range(1, self.degree + 1):
            poly_columns.append(X ** d)

        return np.column_stack(poly_columns)

    def fit(self, X, y):
        """Generates polynomial columns and fits the regression model parameters."""
        X_poly = self._generate_polynomial_features(X)
        super().fit(X_poly, y)
        return self

    def predict(self, X):
        """Generates predictions by expanding new features to match trained dimensions."""
        X_poly = self._generate_polynomial_features(X)
        self._check_is_fitted()
        return super().predict(X_poly)

    def score(self, X, y):
        """Evaluates model performance metrics using raw un-transformed features."""
        return super().score(X, y)



    