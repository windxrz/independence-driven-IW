from stable_learning.methods.baselines import OLS, LASSO
# from stable_learning.methods.dwr import DWR
# from stable_learning.methods.srdo import SRDO


def get_method(method_name):
    return globals()[method_name]
