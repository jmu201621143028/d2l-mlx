import os
import gzip
import random
import time
import numpy as np
import matplotlib.pyplot as plt

import mlx.core as mx

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = mx.array(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def load_mnist_data():
    filename = [
        ["test_images", "t10k-images-idx3-ubyte.gz"], 
        ["test_labels", "t10k-labels-idx1-ubyte.gz"],
        ["train_images", "train-images-idx3-ubyte.gz"], 
        ["train_labels", "train-labels-idx1-ubyte.gz"]
    ]
    mnist = {}
    for key, outfile in filename[0], filename[2]:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outfile = script_dir + "/../data/fashion-mnist/" + outfile
        with gzip.open(outfile, "rb") as f:
            mnist[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                -1, 28, 28, 1
            )
    
    for key, outfile in filename[1], filename[3]:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outfile = script_dir + "/../data/fashion-mnist/" + outfile
        with gzip.open(outfile, "rb") as f:
            mnist[key] = np.frombuffer(f.read(), np.uint8, offset=8)
    return mnist

def show_data(mnist_data):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images, train_labels = mnist_data['train_images'], mnist_data['train_labels']
    N = 12
    plt.figure(figsize=(12, 4))  # 控制整体宽高
    indices = list(range(train_images.shape[0]))
    random.shuffle(indices)
    indices = indices[:N]
    for i, ix in enumerate(indices):
        plt.subplot(2, 6, i + 1)
        image = train_images[ix].squeeze()
        plt.imshow(image, cmap='gray')
        plt.title(class_names[train_labels[ix]], fontsize=10)
        plt.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    plt.show()

start = time.time()
mnist_data = load_mnist_data()
end = time.time()
print(f"loading mnist data time is {end - start}s")
# show_data(mnist_data)

def net(X, w, b):
    X = X.reshape((-1, 784, 1)).squeeze()
    X = X @ w + b
    X_exp = mx.exp(X)
    return X_exp / mx.sum(X_exp, axis=1).reshape((-1, 1))


def cross_entropy(y_hat, y):
    y_hat_correct = mx.take_along_axis(y_hat, y.reshape((-1, 1)), axis=1)
    return mx.mean(-mx.log(y_hat_correct))

def cross_entropy_loss(param, X, y):
    y_hat = net(X, param['w'], param['b'])
    return cross_entropy(y_hat, y)

def cross_entropy_test():
    y = mx.array([0, 2])
    y_hat = mx.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    loss = cross_entropy(y_hat, y)
    return loss
# cross_entropy_test()


batch_size = 64
num_inputs = 784
num_outputs = 10
lr = 0.03
param = {}
param['w'] = mx.random.normal((num_inputs, num_outputs), scale=0.01)
param['b'] = mx.zeros(num_outputs)

loss_and_grad_fn = mx.value_and_grad(cross_entropy_loss)

for X, y in data_iter(batch_size, mnist_data['test_images'], mnist_data['test_labels']):
    X, y = mx.array(X), mx.array(y)
    loss, dloss_para  = loss_and_grad_fn(param, X, y)
    param['w'] -= lr * dloss_para['w']
    param['b'] -= lr * dloss_para['b']
    print(f"loss = {loss}")