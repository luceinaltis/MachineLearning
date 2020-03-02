import numpy as np


# 교차 엔트로피 오차
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# 소프트맥스
def softmax(x):
    c = np.max(x, axis=1)
    c = c.reshape(1, -1).T
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x,axis=1).reshape(1, -1).T
    y = exp_x / sum_exp_x

    return y
