from idiw.methods.baselines import OLS, LASSO

# from idiw.methods.dwr import DWR
# from idiw.methods.srdo import SRDO


def get_method(method_name):
    return globals()[method_name]
