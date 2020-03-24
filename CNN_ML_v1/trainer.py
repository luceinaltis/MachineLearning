import numpy as np
import matplotlib.pyplot as plt
import time

from customed_layers import ConvNet
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True, flatten=False)

#데이터의 크기
x_train, t_train = x_train[:5000], t_train[:5000]

network = ConvNet(input_dim=(1,28,28), 
                conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                hidden_size=100, output_size=10, weight_init_std=0.01)

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []

iter_per_epoch = max(train_size/batch_size, 1)

#코드 시작시간 저장
start = time.time()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print("iter 횟수 : %d, loss : %.8f" % (i, loss))

# 가중치 저장
network.save_params("params.pkl")

print("time : %f초" % (time.time() - start))
plt_x = np.arange(len(train_loss_list))
plt_y = train_loss_list

plt.xlabel("iter")
plt.ylabel("loss volume")

plt.plot(plt_x, plt_y)

plt.show()