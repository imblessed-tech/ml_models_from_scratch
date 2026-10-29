import numpy as np

from linear_models.linear_regression import PolynomialRegressionFromScratch, LinearRegressionFromScratch

X = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000], dtype=float)
X = X.reshape(-1,1)
y = np.array([150000, 165000, 180000, 210000, 220000, 240000, 265000, 270000, 285000, 320000], dtype=float)


for met in ["closed_form", "batch", "stochastic", "mini_batch"]:
    if met == "closed_form":
        pen = None
        linear_cf = LinearRegressionFromScratch(method=met, penalty=pen) 
        linear_cf.fit(X,y)
        r2 = linear_cf.score(X, y)
        print(f"r2 for {met} method and no penalty: {r2}")
    else: 
        for pen in ["ridge", "lasso", "elastic_net", None]:
            linear_cf = LinearRegressionFromScratch(method=met, penalty=pen, n_epochs=1000, batch_size=5, learning_rate=0.01) 
            linear_cf.fit(X,y)
            r2 = linear_cf.score(X, y)
            if pen is None:
                pen = "no"
            print(f"Linear Regression=> r2 for {met} method and {pen} penalty: {r2}")


X_p = np.linspace(-3, 3, 20).reshape(-1, 1)
noise = np.random.normal(0, 0.5, size=(20, 1))
y_p = (0.5 * (X_p ** 2) + 1.0 * X_p + 2.0 + noise).reshape(-1)

for met in ["closed_form", "batch", "stochastic", "mini_batch"]:
    if met == "closed_form":
        pen = None
        ploy_model = PolynomialRegressionFromScratch(method=met,
                                                    degree=2,
                                                    penalty=pen,
                                                    n_epochs=1000, 
                                                    batch_size=5, 
                                                    learning_rate=0.001) 
        ploy_model.fit(X_p,y_p)
        r2 = ploy_model.score(X_p, y_p)
        print(f"Ploynomial Regression=> r2 for {met} method and no penalty: {r2}")
    else: 
        for pen in ["ridge", "lasso", "elastic_net", None]:
            ploy_model = PolynomialRegressionFromScratch(method=met,
                                                        degree=2,
                                                        penalty=pen,
                                                        n_epochs=1000, 
                                                        batch_size=5, 
                                                        learning_rate=0.001) 
            ploy_model.fit(X_p,y_p)
            r2 = ploy_model.score(X_p, y_p)
            if pen is None:
                pen = "no"
            print(f"Ploynomial Regression=> r2 for {met} method and {pen} penalty: {r2}")