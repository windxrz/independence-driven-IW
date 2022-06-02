import numpy as np
from sklearn.linear_model import LinearRegression, Lasso

class OLS:
    def __init__(self, args):
        self.name = "OLS"

    def feature_importance(self, X, y):
        reg = LinearRegression()
        reg.fit(X, y)
        return np.abs(reg.coef_).reshape(-1).tolist()

class LASSO:
    def __init__(self, args):
        self.name = "LASSO_alpha_{}".format(args.alpha)