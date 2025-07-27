import mlx.core as mx
import matplotlib.pyplot as plt
import numpy as np
import random

def synthetic_data(w, b, num_examples):
    X = mx.random.normal((num_examples, len(w)))
    y = X @ w + b
    y += mx.random.normal(y.shape, loc=0, scale=0.01)
    return X, y

true_w = mx.array([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print(f"features.shape = {features.shape}, labels.shape = {labels.shape}")
print(f"features[0] = {features[0]}, labels[0] = {labels[0]}")
# plt.scatter(np.array(features[:, 1]), np.array(labels), s = 2)
# plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = mx.array(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(f"{X.shape}\n{y.shape}")
    break

w = mx.random.normal(shape=(2, 1), scale=0.01)
b = mx.zeros(shape=(1,))


def linreg(X, w, b):
    return X @ w + b

net = linreg
def squared_loss(w, b, X, y):
    y_hat = net(X, w, b)
    return mx.mean(mx.square(y_hat - mx.reshape(y, y_hat.shape))) / 2


def sgd(params, lr, batch_size):
    for param in params:
        param -= lr * param.grad / batch_size
        param.zero_grad()
    pass

lr = 0.03
num_epochs = 3

loss = squared_loss
grad_fn = mx.grad(loss, argnums=(0, 1))
step = 0
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        grad = grad_fn(w, b, X, y)
        dw, db = grad

        w -= lr * dw
        b -= lr * db
        step += 1
        if step % 10 == 0:
            print(f"step {step}, loss = {loss(w, b, X, y)}")

print(f"w = {w}\nb = {b}")