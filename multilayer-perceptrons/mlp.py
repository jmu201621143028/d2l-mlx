import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import numpy as np

import os
import random
import gzip


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
    mnist['train_images'] = mnist['train_images'].astype(np.float32) / 255.0
    mnist['test_images'] = mnist['test_images'].astype(np.float32) / 255.0
    return mnist

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
    
        ]
    def __call__(self, x):
        x = x.reshape((-1, 784, 1)).squeeze()
        for l in self.layers:
            x = l(x)
        return x

def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

def eval_fn(model, X, y):
    X, y = mx.array(X), mx.array(y)
    return mx.mean(mx.argmax(model(X), axis=1) == y)


mnist_data = load_mnist_data()
batch_size = 128
num_inputs = 784
num_outputs = 10
lr = 0.01
num_epochs = 500
model = MLP()
mx.eval(model.parameters())

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.SGD(learning_rate=lr)


for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    for X, y in data_iter(batch_size, mnist_data['train_images'], mnist_data['train_labels']):
        X, y = mx.array(X), mx.array(y)
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        epoch_loss += loss
        num_batches += 1
    accuracy = eval_fn(model, mnist_data['test_images'], mnist_data['test_labels'])
    print(f"Epoch {epoch}: loss = {epoch_loss / num_batches:.4f}, Test accuracy {accuracy.item():.3f}")