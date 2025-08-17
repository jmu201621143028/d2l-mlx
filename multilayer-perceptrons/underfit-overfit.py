import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import matplotlib.pyplot as plt

import random

def gen_data():
    max_degree = 20
    n_train, n_test = 100, 100
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    '''
        features = (x1; x2; ... xn)
        poly_features = (x1**0, x1**1, x1**2, ... x1**19;
                         x2**0, x2**1, x2**2, ... x2**19;
                         ...)
    '''
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1) # gamma(n) = (n - 1)!
    labels = np.dot(poly_features, true_w)
    labels += np.random.normal(scale=1.0, size=labels.shape)
    return poly_features, labels

X, Y = gen_data()
print(X[:2])
print(Y[:2])

class LinearNet(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.layers = [nn.Linear(in_dims, 1)]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

def loss_fn(model, X, y):
    y = y.reshape((-1, 1))
    return nn.losses.mse_loss(model(X), y)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = mx.array(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def train(train_features, test_features, train_labels, test_labels,
          num_epochs = 400):
    input_shape = train_features.shape[-1]
    batch_size = 10
    net = LinearNet(input_shape)
    loss_and_grad_fn = nn.value_and_grad(net, loss_fn)
    optimizer = optim.SGD(learning_rate=0.05)
    px = np.arange(0, num_epochs, 1)
    py, py_test = [], []
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        for X, y in data_iter(batch_size, train_features, train_labels):
            X, y = mx.array(X), mx.array(y)
            loss, grads = loss_and_grad_fn(net, X, y)
            optimizer.update(net, grads)
            epoch_loss += float(loss.item())
            num_batches += 1
        print(f"Epoch {epoch}: loss = {epoch_loss / num_batches:.4f}")
        loss, _ = loss_and_grad_fn(net, mx.array(test_features), mx.array(test_labels))
        py.append(epoch_loss / num_batches)
        py_test.append(float(loss.item()))
    plt.plot(px, py, label="train loss")
    plt.plot(px, py_test, label="test loss")
    plt.legend()

    
n_train = 100

plt.subplot(3, 1, 1)
plt.title("normal")
train(X[:n_train, :4], X[n_train:, :4],
      Y[:n_train], Y[n_train:])

plt.subplot(3, 1, 2)
plt.title("underfit")
train(X[:n_train, :2], X[n_train:, :2],
      Y[:n_train], Y[n_train:])

plt.subplot(3, 1, 3)
plt.title("overfit")
train(X[:n_train, :], X[n_train:, :],
      Y[:n_train], Y[n_train:])
plt.tight_layout()
plt.show()
